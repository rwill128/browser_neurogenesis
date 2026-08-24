# Legacy code location

Legacy app paths were moved out of this repo tree to:

- `/Users/richardwilliams/browser_neurogenesis_legacy_code/`

Moved items:
- `js/`
- `css/`
- `index.html`
- `edit_creature.html`

Compatibility strategy (no symlinks):
- `js/` now contains lightweight bridge modules that re-export from the external legacy directory.
- `index.html` and `edit_creature.html` are lightweight pointer pages.
- `css/style.css` is a lightweight pointer placeholder.
- `sim-server/public/full-sim/` contains the static publication snapshot used by GitHub Pages; the external directory remains the editable source of truth.
