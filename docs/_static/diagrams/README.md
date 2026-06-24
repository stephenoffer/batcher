# Documentation diagrams

The architecture diagrams used in the docs are generated from Graphviz, not drawn by
hand, so they stay consistent and easy to update.

- **`render.py`** is the single source of truth — it defines every diagram and a
  shared style, then shells out to `dot` to render each `<name>.png` here.
- **`*.png`** are committed so the Sphinx build (and the GitHub Pages workflow) needs
  no Graphviz. The intermediate `*.dot` files are generated and git-ignored.

## Regenerating

```bash
brew install graphviz      # provides `dot` (one-time)
just diagrams              # or: python docs/_static/diagrams/render.py
```

Then rebuild the docs (`just docs`) and commit the updated PNGs.

## Adding or editing a diagram

Edit `render.py`: add a `@diagram("name")` function returning a Graphviz body, or
change an existing one. Reference it from a Markdown page with a relative path and
descriptive alt text, e.g.

```markdown
![Alt text describing the diagram](../_static/diagrams/name.png)
```
