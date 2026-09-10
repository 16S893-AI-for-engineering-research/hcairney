// @ts-check
import { defineConfig } from 'astro/config';

import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  site: 'https://16s893-ai-for-engineering-research.github.io',
  base: '/hcairney/',
  vite: {
    plugins: [tailwindcss()]
  }
});
