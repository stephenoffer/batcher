{# Sphinx's stock autosummary class template opens the body with an explicit
   `.. automethod:: __init__`. conf.py already sets `special-members: __init__` in
   `autodoc_default_options`, so `autoclass` documents the constructor on its own and
   the explicit directive documents it a second time -- a `duplicate object
   description` warning, which the -W build treats as an error. Dropping the line is
   the whole difference from the stock template. #}
{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
