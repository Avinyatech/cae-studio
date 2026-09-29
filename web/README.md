# Web

A single-file web app (`index.html` — no build step) for the design
studio: parametric shell analysis and uploaded-geometry (STL → tet mesh)
workflows, both with interactive 3D viewers.

## Running it

Open `index.html` directly in a browser, or publish it anywhere static
files are served. It talks to the backend at the address shown in the
top-right field (defaults to `http://127.0.0.1:8000`, editable in the
page).

If you publish this page somewhere over HTTPS (e.g. as a Claude Artifact)
while running the backend locally, point the address field at
`http://127.0.0.1:8000` specifically, not a LAN IP — browsers allow an
HTTPS page to call `localhost`/`127.0.0.1` over plain HTTP as a
mixed-content exception, but not an arbitrary LAN address.

## What's in the page

- **Parametric workflow**: length/width/thickness/mesh-density form,
  material and boundary-condition pickers, three analysis types (normal
  modes, static, frequency response), animated 3D-isometric SVG viewers,
  and a stress contour panel (static analysis).
- **Upload workflow**: STL upload, a real Three.js 3D viewer (loaded from
  jsdelivr — CSP-compliant if you publish this as a Claude Artifact),
  click-to-pick faces for fixed supports and load points, then the same
  three analysis types solved on the actual mesh.
- **Local LLM assistant panel**: describe a structure in plain English to
  fill the form, or get a grounded explanation when a solve fails.

No external dependencies beyond Google Fonts and the Three.js/OrbitControls
CDN scripts — everything else is inline.
