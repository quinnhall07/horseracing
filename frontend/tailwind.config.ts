import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "ui-monospace", "monospace"],
      },
      fontSize: {
        xxs: ["0.6875rem", { lineHeight: "1rem" }],
      },
      colors: {
        edge: {
          pos: "#10b981", // emerald-500
          neg: "#f43f5e", // rose-500
        },
      },
    },
  },
  plugins: [],
};

export default config;
