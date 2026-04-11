import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./hooks/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#070b14",
        panel: "#0e1628",
        line: "rgba(148, 163, 184, 0.2)",
        text: "#e5edff",
        muted: "#9db0d1",
        accent: "#4f8cff"
      },
      boxShadow: {
        glow: "0 10px 30px rgba(79, 140, 255, 0.2)"
      }
    }
  },
  plugins: [],
};

export default config;
