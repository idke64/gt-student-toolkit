// @ts-check
import { defineConfig } from "astro/config";

import tailwind from "@astrojs/tailwind";
import react from "@astrojs/react";
import path from "node:path";

// https://astro.build/config
export default defineConfig({
  output: "server",
  integrations: [react(), tailwind({})],

  vite: {
    plugins: [tailwind()],
    resolve: {
      alias: {
        "@components": path.resolve("./src/components"),
        "@layouts": path.resolve("./src/layouts"),
        "@styles": path.resolve("./src/styles"),
        "@pages": path.resolve("./src/pages"),
      },
    },
  },
});
