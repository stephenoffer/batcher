{#-
  Table and view, replaced in place with CREATE OR REPLACE. The default materializations
  build an intermediate relation and rename it over the target, and Batcher has no rename.
  A relation of the other kind under the same name is dropped first.
-#}

{% materialization table, adapter='batcher' %}
  {%- set target_relation = this.incorporate(type='table') -%}
  {%- set existing = load_cached_relation(this) -%}
  {{ run_hooks(pre_hooks) }}
  {% if existing is not none and not existing.is_table %}
    {{ adapter.drop_relation(existing) }}
  {% endif %}
  {% call statement('main') -%}
    {{ get_create_table_as_sql(False, target_relation, sql) }}
  {%- endcall %}
  {{ run_hooks(post_hooks) }}
  {{ return({'relations': [target_relation]}) }}
{% endmaterialization %}

{% materialization view, adapter='batcher' %}
  {%- set target_relation = this.incorporate(type='view') -%}
  {%- set existing = load_cached_relation(this) -%}
  {{ run_hooks(pre_hooks) }}
  {% if existing is not none and not existing.is_view %}
    {{ adapter.drop_relation(existing) }}
  {% endif %}
  {% call statement('main') -%}
    {{ get_create_view_as_sql(target_relation, sql) }}
  {%- endcall %}
  {{ run_hooks(post_hooks) }}
  {{ return({'relations': [target_relation]}) }}
{% endmaterialization %}
