module.exports = {
  content: [
    "./src/**/*.{astro,html,js,jsx,ts,tsx,svelte,css}",
    "./public/**/*.html",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          100: "rgb(var(--primary-100) / <alpha-value>)",
          200: "rgb(var(--primary-200) / <alpha-value>)",
          300: "rgb(var(--primary-300) / <alpha-value>)",
        },
        surface: {
          100: "rgb(var(--surface-100) / <alpha-value>)",
          200: "rgb(var(--surface-200) / <alpha-value>)",
          300: "rgb(var(--surface-300) / <alpha-value>)",
        },
        border: "rgb(var(--border) / <alpha-value>)",
        fg: {
          100: "rgb(var(--fg-100) / <alpha-value>)",
          200: "rgb(var(--fg-200) / <alpha-value>)",
          300: "rgb(var(--fg-300) / <alpha-value>)",
          400: "rgb(var(--fg-400) / <alpha-value>)",
        },
        themeicon: "rgb(var(--themeicon) / <alpha-value>)",
        "btn-hover": "rgb(var(--btn-hover) / <alpha-value>)",
      },
      boxShadow: {
        DEFAULT: "0 2px 8px rgba(0, 0, 0, 0.1)",
      },
    },
  },
  plugins: [],
};