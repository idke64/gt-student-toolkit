// @ts-check
import { defineConfig } from "astro/config";

import react from "@astrojs/react";
import tailwind from "@astrojs/tailwind";
import path from "node:path";

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
        "@firebase": path.resolve("./src/firebase"),
      },
    },
  },
});
