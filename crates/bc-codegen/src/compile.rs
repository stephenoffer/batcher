//! Build and JIT-compile a Cranelift function that evaluates an `Expr`
//! element-wise over the row index, returning the finalized function pointer.

use std::collections::HashMap;

use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::emit::Codegen;
use crate::simd::SimdCodegen;
use crate::{CodegenError, ColumnSet, Compiled, ScalarTy};

/// libm symbols for the single-arg math functions the JIT lowers via a libcall.
/// These are exactly the symbols Rust's `f64` methods lower to on this platform,
/// so the JIT result is bit-for-bit identical to the interpreter's `eval_math`.
const LIBM_UNARY: &[&str] = &[
    "log", "log10", "log2", "exp", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos",
    "atan",
];

/// libm symbols for the two-arg math functions the JIT lowers via a libcall.
const LIBM_BINARY: &[&str] = &["pow", "atan2"];

/// Emit a read-modify-write that packs one boolean (i8 `0`/`1` in `val`) into the
/// Arrow bitmask at row `i`: `out[i >> 3] |= (val & 1) << (i & 7)`. The output
/// buffer is zero-initialized by `eval`, and each loop is sequential, so OR-ing
/// distinct bits into a shared byte is correct — including where the SIMD body and
/// the scalar remainder both touch the trailing byte (the remainder's RMW observes
/// the bits the vector body already set, because the vector loop runs to completion
/// first).
fn emit_bool_bit_store(b: &mut FunctionBuilder, out_ptr: Value, i: Value, val: Value) {
    let byte_idx = b.ins().ushr_imm(i, 3); // i / 8
    let addr = b.ins().iadd(out_ptr, byte_idx);
    let bit = b.ins().band_imm(i, 7); // i % 8 (Arrow packs LSB-first)
    let lo = b.ins().band_imm(val, 1); // defensively mask to 0/1
    let shifted = b.ins().ishl(lo, bit);
    let cur = b.ins().load(types::I8, MemFlags::trusted(), addr, 0);
    let new = b.ins().bor(cur, shifted);
    b.ins().store(MemFlags::trusted(), new, addr, 0);
}

/// Build and JIT-compile a function evaluating `expr` element-wise.
///
/// `kleene` selects the ABI. The default value-only ABI is
/// `(n, cols: *const *const u8, out: *mut u8)`. The Kleene ABI adds a parallel
/// per-column validity pointer array and a validity output buffer —
/// `(n, cols, nulls: *const *const u8, out, valid: *mut u8)` — so `And`/`Or` can
/// compute a correct three-valued-logic validity the simple combined-mask can't.
pub(crate) fn compile(
    expr: &bc_expr::Expr,
    cols: &ColumnSet,
    result_ty: ScalarTy,
    kleene: bool,
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
    if kleene {
        sig.params.push(AbiParam::new(ptr_ty)); // nulls: *const *const u8
    }
    sig.params.push(AbiParam::new(ptr_ty)); // out ptr
    if kleene {
        sig.params.push(AbiParam::new(ptr_ty)); // valid out ptr
    }

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
        // The validity-array base and validity-output pointer only exist in the
        // Kleene ABI; the value-only ABI keeps its original 3-param layout.
        let (null_base, out_ptr, valid_ptr) = if kleene {
            (Some(params[2]), params[3], Some(params[4]))
        } else {
            (None, params[2], None)
        };

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
        // The parallel per-column validity base pointers (Kleene mode only).
        let null_ptrs: Option<Vec<Value>> = null_base.map(|nb| {
            (0..cols.order.len() as i32)
                .map(|k| b.ins().load(ptr_ty, MemFlags::trusted(), nb, k * ptr_bytes))
                .collect()
        });

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
            null_ptrs: null_ptrs.as_deref(),
            i,
            fmod: fmod_ref,
            libm: &libm,
        };
        let val = gen.emit(expr);
        // In Kleene mode also compute the row's validity (both via `gen`, before its
        // borrow of `b` ends). A Kleene expression is always boolean (it contains
        // And/Or in boolean position), so its value buffer is the 1-byte ABI.
        let valid = if kleene {
            Some(gen.emit_validity(expr))
        } else {
            None
        };
        // Store the result. The value-only boolean path packs one bit per row into
        // an Arrow bitmask (see `emit_bool_bit_store`) so `eval` wraps the buffer
        // with no per-element repack — the dominant cost when the output is boolean.
        // The Kleene path keeps the 1-byte `out`/`valid` ABI (`eval_kleene` reads
        // bytes); i64/f64 store an 8-byte word.
        if result_ty == ScalarTy::Bool && !kleene {
            emit_bool_bit_store(&mut b, out_ptr, i, val);
        } else {
            let elem_bytes = match result_ty {
                ScalarTy::Bool => 1i64,
                // Temporal results are guarded out in `compile_expr`, so this width
                // is never actually used for one; 8 keeps the match total.
                ScalarTy::I64 | ScalarTy::F64 | ScalarTy::Date32 | ScalarTy::TsUs => 8i64,
            };
            let off = b.ins().imul_imm(i, elem_bytes);
            let addr = b.ins().iadd(out_ptr, off);
            b.ins().store(MemFlags::trusted(), val, addr, 0);
            if let (Some(valid), Some(valid_ptr)) = (valid, valid_ptr) {
                // Validity is i8, one byte per row.
                let vaddr = b.ins().iadd(valid_ptr, i);
                b.ins().store(MemFlags::trusted(), valid, vaddr, 0);
            }
        }
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

/// Build and JIT-compile a **vectorized** function for the `simd_ty` subset
/// (numeric arithmetic, comparisons, `Not`, exact casts). Same value-only ABI as
/// the scalar path (`(n, cols, out)`), so it is a drop-in for it — `eval` and the
/// null-mask handling are unchanged.
///
/// `lanes` (2/4/8) is the f64/i64 vector width and `unroll` (≥ 1) the number of
/// independent vector chains emitted per iteration (instruction-level parallelism),
/// both chosen from the host [`HardwareProfile`](bc_arrow::HardwareProfile). The
/// vector loop covers the largest multiple of `lanes*unroll` rows ≤ `n`, then a
/// scalar remainder loop ([`Codegen`]) handles the tail one row at a time. The
/// vector ops are per-lane identical to the scalar ops, so the result stays
/// bit-for-bit equal to the interpreter oracle at any width/unroll.
///
/// `result_ty` selects how a row is stored: an `I64`/`F64` result is a `lanes*8`-byte
/// vector store per chain; a `Bool` result's `I64xL` canonical mask is packed one bit
/// per row into the Arrow bitmask (see [`emit_bool_bit_store`] for the tail).
pub(crate) fn compile_simd(
    expr: &bc_expr::Expr,
    cols: &ColumnSet,
    result_ty: ScalarTy,
    lanes: usize,
    unroll: usize,
) -> Result<Compiled, CodegenError> {
    use crate::simd::vec_ty;
    use cranelift_codegen::ir::condcodes::IntCC;
    debug_assert!(matches!(lanes, 2 | 4 | 8) && unroll >= 1);
    let step = (lanes * unroll) as i64;

    let mut flag_builder = settings::builder();
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

    // `fmod` is declared so the scalar remainder can reuse `Codegen` unchanged; the
    // SIMD subset (`+,-,*,/`) never calls it.
    let mut fmod_sig = module.make_signature();
    fmod_sig.params.push(AbiParam::new(types::F64));
    fmod_sig.params.push(AbiParam::new(types::F64));
    fmod_sig.returns.push(AbiParam::new(types::F64));
    let fmod_id = module
        .declare_function("fmod", Linkage::Import, &fmod_sig)
        .map_err(|e| CodegenError::Cranelift(e.to_string()))?;

    let mut ctx = module.make_context();
    let ptr_ty = module.target_config().pointer_type();
    let sig = &mut ctx.func.signature;
    sig.params.push(AbiParam::new(types::I64)); // n
    sig.params.push(AbiParam::new(ptr_ty)); // cols: *const *const u8
    sig.params.push(AbiParam::new(ptr_ty)); // out: *mut u8 (f64)

    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let entry = b.create_block();
        let vheader = b.create_block();
        let vbody = b.create_block();
        let rheader = b.create_block();
        let rbody = b.create_block();
        let exit = b.create_block();

        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let params: Vec<Value> = b.block_params(entry).to_vec();
        let n = params[0];
        let cols_base = params[1];
        let out_ptr = params[2];

        let ptr_bytes = ptr_ty.bytes() as i32;
        let col_ptrs: Vec<Value> = (0..cols.order.len() as i32)
            .map(|k| {
                b.ins()
                    .load(ptr_ty, MemFlags::trusted(), cols_base, k * ptr_bytes)
            })
            .collect();
        // Unaligned vector flags: the output buffer is only 8-byte aligned, so the
        // engine never asserts wider vector alignment for these stores.
        let vflags = MemFlags::new().with_notrap();
        let bool_result = result_ty == ScalarTy::Bool;
        // The vector mask's lane type when the result is boolean (`I64xL`).
        let mask_ty = vec_ty(ScalarTy::I64, lanes);

        // `main` = largest multiple of `step` ≤ n; the vector loop covers `[0, main)`.
        // (`step` need not be a power of two — `unroll` can be any ≥ 1 — so compute it
        // as `n - n % step` rather than a bit-mask.)
        let i_ty = types::I64;
        let rem = b.ins().urem_imm(n, step);
        let main = b.ins().isub(n, rem);

        // --- vector loop (`unroll` chains of `lanes` rows per iteration) ---
        b.append_block_param(vheader, i_ty);
        let zero = b.ins().iconst(i_ty, 0);
        b.ins().jump(vheader, &[zero.into()]);
        b.switch_to_block(vheader);
        let vi = b.block_params(vheader)[0];
        let vcond = b.ins().icmp(IntCC::SignedLessThan, vi, main);
        b.ins().brif(vcond, vbody, &[], rheader, &[main.into()]);

        b.switch_to_block(vbody);
        for c in 0..unroll {
            // This chain covers rows `[base, base + lanes)`.
            let base = if c == 0 {
                vi
            } else {
                b.ins().iadd_imm(vi, (c * lanes) as i64)
            };
            let vval = {
                let mut gen = SimdCodegen {
                    b: &mut b,
                    cols,
                    col_ptrs: &col_ptrs,
                    i: base,
                    lanes,
                };
                gen.emit(expr)
            };
            if bool_result {
                // Pack this chain's `lanes` canonical-mask lanes (one bit each) into
                // the Arrow bitmask byte `out[base >> 3]` via one RMW. `base` is a
                // multiple of `lanes` and `lanes <= 8`, so all `lanes` bits fall in a
                // single byte; the loop is sequential, so OR-ing distinct bits (across
                // chains/iterations that share a byte, and with the scalar remainder)
                // is correct over the zero-initialized buffer.
                let one = b.ins().iconst(types::I64, 1);
                let one_vec = b.ins().splat(mask_ty, one);
                let bits = b.ins().band(vval, one_vec);
                let bit0 = b.ins().band_imm(base, 7); // base % 8
                let mut combined = b.ins().iconst(types::I8, 0);
                for l in 0..lanes {
                    let lane = b.ins().extractlane(bits, l as u8);
                    let bl = b.ins().ireduce(types::I8, lane);
                    let shift = b.ins().iadd_imm(bit0, l as i64);
                    let sl = b.ins().ishl(bl, shift);
                    combined = b.ins().bor(combined, sl);
                }
                let byte_idx = b.ins().ushr_imm(base, 3);
                let baddr = b.ins().iadd(out_ptr, byte_idx);
                let cur = b.ins().load(types::I8, MemFlags::trusted(), baddr, 0);
                let new = b.ins().bor(cur, combined);
                b.ins().store(MemFlags::trusted(), new, baddr, 0);
            } else {
                // I64/F64: a single `lanes*8`-byte vector store at row `base`.
                let voff = b.ins().imul_imm(base, 8);
                let vaddr = b.ins().iadd(out_ptr, voff);
                b.ins().store(vflags, vval, vaddr, 0);
            }
        }
        let vnext = b.ins().iadd_imm(vi, step);
        b.ins().jump(vheader, &[vnext.into()]);

        // --- scalar remainder loop (the up-to-`step-1` tail rows) ---
        b.append_block_param(rheader, i_ty);
        b.switch_to_block(rheader);
        let ri = b.block_params(rheader)[0];
        let rcond = b.ins().icmp(IntCC::SignedLessThan, ri, n);
        b.ins().brif(rcond, rbody, &[], exit, &[]);

        b.switch_to_block(rbody);
        let fmod_ref = module.declare_func_in_func(fmod_id, b.func);
        let libm = std::collections::HashMap::new();
        let rval = {
            let mut gen = Codegen {
                b: &mut b,
                cols,
                col_ptrs: &col_ptrs,
                null_ptrs: None,
                i: ri,
                fmod: fmod_ref,
                libm: &libm,
            };
            gen.emit(expr)
        };
        // A boolean result packs the trailing row's bit into the same Arrow bitmask
        // the vector body wrote (RMW preserves the bits already set); i64/f64 store
        // an 8-byte word. The scalar `Codegen` returns an i8 0/1 for a comparison.
        if result_ty == ScalarTy::Bool {
            emit_bool_bit_store(&mut b, out_ptr, ri, rval);
        } else {
            let roff = b.ins().imul_imm(ri, 8);
            let raddr = b.ins().iadd(out_ptr, roff);
            b.ins().store(MemFlags::trusted(), rval, raddr, 0);
        }
        let rnext = b.ins().iadd_imm(ri, 1);
        b.ins().jump(rheader, &[rnext.into()]);

        b.seal_block(vheader);
        b.seal_block(vbody);
        b.seal_block(rheader);
        b.seal_block(rbody);
        b.switch_to_block(exit);
        b.seal_block(exit);
        b.ins().return_(&[]);
        b.finalize();
    }

    let func_id = module
        .declare_function("jit_simd_expr", Linkage::Export, &ctx.func.signature)
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
