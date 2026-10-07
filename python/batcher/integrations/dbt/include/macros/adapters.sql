{#- The SQL the Batcher adapter emits. Plain Jinja macros, so tests render them without dbt. -#}

{% macro batcher__create_table_as(temporary, relation, compiled_code, language='sql') -%}
  create or replace table {{ relation }} as ({{ compiled_code }})
{%- endmacro %}

{% macro batcher__create_view_as(relation, sql) -%}
  create or replace view {{ relation }} as ({{ sql }})
{%- endmacro %}

{% macro batcher__current_timestamp() -%}
  current_timestamp
{%- endmacro %}
