# Mobile

Expo / React Native client for the same backend API, for running
parametric design studies from a phone on the same network as the
backend.

## Running it

```bash
npm install
npx expo start
```

Scan the QR code with Expo Go, or run on a simulator. Set the API address
in the app to your backend's LAN IP (e.g. `http://192.168.x.x:8000`) — a
phone isn't `localhost` relative to the machine running the backend.

## Structure

- `App.tsx` — the whole UI: parameter form, results, mode-shape viewer
- `src/api.ts` — typed fetch client for the backend
- `src/modeViewerHtml.ts` — generates the isometric 3D mode-shape/static
  deflection viewer as an HTML string, rendered in a `WebView`
