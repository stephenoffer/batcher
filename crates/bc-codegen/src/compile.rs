//! Build and JIT-compile a Cranelift function that evaluates an `Expr`
//! element-wise over the row index, returning the finalized function pointer.

use std::collections::HashMap;

use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::emit::Codegen;
use crate::{CodegenError, ColumnSet, Compiled, ScalarTy};

/// libm symbols for the single-arg math functions the JIT lowers via a libcall.
/// These are exactly the symbols Rust's `f64` methods lower to on this platform,
/// so the JIT result is bit-for-bit identical to the interpreter's `eval_math`.
const LIBM_UNARY: &[&str] = &[
    "log", "log10", "log2", "exp", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos",
    "atan", "cbrt",
];

/// libm symbols for the two-arg math functions the JIT lowers via a libcall.
const LIBM_BINARY: &[&str] = &["pow", "atan2"];

/// Build and JIT-compile a function evaluating `expr` element-wise.
pub(crate) fn compile(
    expr: &bc_expr::Expr,
    cols: &ColumnSet,
    result_ty: ScalarTy,
) -> Result<Compiled, CodegenError> {
    let mut flag_builder = settings::builder();
    // `is_pic` keeps relocations relative; opt for speed of the generated body.
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| CodegenError::Cranelift(e.to_string()))?;
    let isa_builder =
        cranelift_native::builder().map_err(|e| CodegenError::Cranelift(e.to_string()))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| CodegenError::Cranelift(e.to_string()))?;

    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);

    // Declare libc `fmod` so float `%` matches Arrow's `f64 % f64` (which lowers
    // to `fmod`) bit-for-bit. The JIT resolves the symbol from the process.
    let mut fmod_sig = module.make_signature();
    fmod_sig.params.push(AbiParam::new(types::F64));
    fmod_sig.params.push(AbiParam::new(types::F64));
    fmod_sig.returns.push(AbiParam::new(types::F64));
    let fmod_id = module
        .declare_function("fmod", Linkage::Import, &fmod_sig)
        .map_err(|e| CodegenError::Cranelift(e.to_string()))?;

    // Declare the libm transcendentals / two-arg math we lower to. Rust's `f64`
    // methods (`ln`, `sin`, `powf`, ...) lower to exactly these symbols on this
    // platform, so the JIT call is bit-for-bit identical to the interpreter; the
    // `differential_transcendental` test confirms parity per function. We share
    // one (f64)->f64 signature and one (f64,f64)->f64 signature for all of them.
    let mut unary_sig = module.make_signature();
    unary_sig.params.push(AbiParam::new(types::F64));
    unary_sig.returns.push(AbiParam::new(types::F64));
    let mut binary_sig = module.make_signature();
    binary_sig.params.push(AbiParam::new(types::F64));
    binary_sig.params.push(AbiParam::new(types::F64));
    binary_sig.returns.push(AbiParam::new(types::F64));

    // Every libm symbol used by `emit_typed`, keyed by name so `Codegen` can look
    // up the `FuncRef` for the function it is lowering.
    let mut libm_ids: HashMap<&'static str, cranelift_module::FuncId> = HashMap::new();
    for &name in LIBM_UNARY {
        let id = module
            .declare_function(name, Linkage::Import, &unary_sig)
            .map_err(|e| CodegenError::Cranelift(e.to_string()))?;
        libm_ids.insert(name, id);
    }
    for &name in LIBM_BINARY {
        let id = module
            .declare_function(name, Linkage::Import, &binary_sig)
            .map_err(|e| CodegenError::Cranelift(e.to_string()))?;
        libm_ids.insert(name, id);
    }

    let mut ctx = module.make_context();
    let ptr_ty = module.target_config().pointer_type();

    // Signature: (n: i64, cols: *const *const u8, out: *mut u8). Columns arrive as
    // one pointer array (not one param each), so the supported column count is
    // unbounded — each base pointer is loaded from the array at entry below.
    let sig = &mut ctx.func.signature;
    sig.params.push(AbiParam::new(types::I64)); // n
    sig.params.push(AbiParam::new(ptr_ty)); // cols: *const *const u8
    sig.params.push(AbiParam::new(ptr_ty)); // out ptr

    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry = b.create_block();
        let header = b.create_block();
        let body = b.create_block();
        let exit = b.create_block();

        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);

        let params: Vec<Value> = b.block_params(entry).to_vec();
        let n = params[0];
        let cols_base = params[1];
        let out_ptr = params[2];

        // Load each column's base pointer from the pointer array once (in `entry`,
        // so the loads are loop-invariant). `col_ptrs[k]` then means exactly what it
        // did under the per-param ABI, so `emit` is unchanged.
        let ptr_bytes = ptr_ty.bytes() as i32;
        let col_ptrs: Vec<Value> = (0..cols.order.len() as i32)
            .map(|k| {
                b.ins()
                    .load(ptr_ty, MemFlags::trusted(), cols_base, k * ptr_bytes)
            })
            .collect();

        // Loop index `i`, threaded as a block parameter on `header`.
        let i_ty = types::I64;
        b.append_block_param(header, i_ty);
        let zero = b.ins().iconst(i_ty, 0);
        b.ins().jump(header, &[zero.into()]);

        // header: if i < n goto body else exit
        b.switch_to_block(header);
        let i = b.block_params(header)[0];
        let cond = b.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
            i,
            n,
        );
        b.ins().brif(cond, body, &[], exit, &[]);

        // body: out[i] = expr(cols[i]); i += 1; goto header
        b.switch_to_block(body);
        let fmod_ref = module.declare_func_in_func(fmod_id, b.func);
        let libm: HashMap<&'static str, cranelift_codegen::ir::FuncRef> = libm_ids
            .iter()
            .map(|(&name, &id)| (name, module.declare_func_in_func(id, b.func)))
            .collect();
        let mut gen = Codegen {
            b: &mut b,
            cols,
            col_ptrs: &col_ptrs,
            i,
            fmod: fmod_ref,
            libm: &libm,
        };
        let val = gen.emit(expr);
        // Store to out[i]; element width follows the result scalar type (1 byte
        // for the u8 boolean ABI, 8 bytes for i64/f64). The store's type is
        // inferred from `val`, which has scalar type `result_ty`.
        let elem_bytes = match result_ty {
            ScalarTy::Bool => 1i64,
            ScalarTy::I64 | ScalarTy::F64 => 8i64,
        };
        let off = b.ins().imul_imm(i, elem_bytes);
        let addr = b.ins().iadd(out_ptr, off);
        b.ins().store(MemFlags::trusted(), val, addr, 0);
        let one = b.ins().iconst(i_ty, 1);
        let next = b.ins().iadd(i, one);
        b.ins().jump(header, &[next.into()]);

        b.seal_block(header);
        b.seal_block(body);

        b.switch_to_block(exit);
        b.seal_block(exit);
        b.ins().return_(&[]);

        b.finalize();
    }

    let func_id = module
        .declare_function("jit_expr", Linkage::Export, &ctx.func.signature)
        .map_err(|e| CodegenError::Cranelift(e.to_string()))?;
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| CodegenError::Cranelift(e.to_string()))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| CodegenError::Cranelift(e.to_string()))?;

    let ptr = module.get_finalized_function(func_id);
    Ok(Compiled {
        ptr,
        nargs: cols.order.len(),
        _module: module,
    })
}
