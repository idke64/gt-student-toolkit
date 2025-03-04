module.exports = {
  content: [
    "./src/**/*.{astro,html,js,jsx,ts,tsx,svelte,css}",
    "./public/**/*.html",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          100: "rgba(var(--primary-100))",
          200: "rgba(var(--primary-200))",
          300: "rgba(var(--primary-300))",
        },
        themeicon: "rgba(var(--themeicon))",
        border: "rgba(var(--border))",
        "btn-hover": "rgba(var(--btn-hover))",
        surface: {
          100: "rgba(var(--surface-100))",
          200: "rgba(var(--surface-200))",
          300: "rgba(var(--surface-300))",
        },
        fg: {
          100: "rgba(var(--fg-100))",
          200: "rgba(var(--fg-200))",
          300: "rgba(var(--fg-300))",
          400: "rgba(var(--fg-400))",
        },
      },
      boxShadow: {
        DEFAULT: "0 2px 8px rgba(0, 0, 0, 0.1)",
      },
    },
  },
  plugins: [],
};
