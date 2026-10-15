import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// This file runs in Node, but the project does not depend on @types/node and does not need to for one
// lookup. Declaring just the shape used here keeps the typecheck honest without adding a dependency.
declare const process: { env: Record<string, string | undefined> };

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) {
            return undefined;
          }

          if (
            /[\\/]node_modules[\\/](react-markdown|remark-|rehype-|micromark|mdast-|hast-|unified|unist-|vfile|property-information|space-separated-tokens|comma-separated-tokens|decode-named-character-reference|character-entities|trim-lines|trough|bail|ccount|devlop)/u.test(
              id
            )
          ) {
            return "markdown";
          }

          return undefined;
        }
      }
    }
  },
  server: {
    // strictPort stays on so the app never silently moves to a port the API's CORS origin does not expect,
    // but the port itself is configurable: 5173 is a busy default and a developer with another Vite project
    // running could not start this one at all.
    port: Number(process.env.DYNAMICCHAT_WEB_PORT ?? process.env.PORT ?? 5173),
    strictPort: true,
    // No CORS at all. The app's page is served by this server, so everything it loads from here is same-origin; the
    // API server answers its own CORS. Vite's default allows every loopback origin, and on a machine running the
    // photoreal backend that includes ComfyUI's web UI (http://127.0.0.1:8188), where any installed custom node runs
    // its own JavaScript — the same origin the API server refuses on every guarded route.
    cors: false,
    fs: {
      // Vite's defaults — a list given here replaces them rather than adding to them — plus the data directory. The
      // default DYNAMICCHAT_DATA_DIR is inside this project, which is the folder Vite serves: without this entry
      // /.dynamicchat-data/dev-secrets.json (the stored API keys, base64 only) and state.json (every transcript)
      // were served like any other file. A data directory moved outside the project is outside fs.allow already;
      // one moved somewhere else inside it needs its own entry here.
      deny: [".env", ".env.*", "*.{crt,pem}", "**/.git/**", "**/.dynamicchat-data/**"]
    }
  }
});
