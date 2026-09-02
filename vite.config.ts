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
    strictPort: true
  }
});
