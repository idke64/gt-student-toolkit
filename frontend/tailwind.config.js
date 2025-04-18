module.exports = {
  content: [
    "./src/**/*.{astro,html,js,jsx,ts,tsx,svelte,css}",
    "./public/**/*.html",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          100: "rgb(var(--primary-100))",
          200: "rgb(var(--primary-200))",
          300: "rgb(var(--primary-300))",
        },
        surface: {
          100: "rgb(var(--surface-100))",
          200: "rgb(var(--surface-200))",
          300: "rgb(var(--surface-300))",
        },
        border: "rgb(var(--border))",
        fg: {
          100: "rgb(var(--fg-100))",
          200: "rgb(var(--fg-200))",
          300: "rgb(var(--fg-300))",
          400: "rgb(var(--fg-400))",
        },
        themeicon: "rgb(var(--themeicon))",
        "btn-hover": "rgb(var(--btn-hover))",
      },
      boxShadow: {
        DEFAULT: "0 2px 8px rgba(0, 0, 0, 0.1)",
      },
    },
  },
  plugins: [],
};
