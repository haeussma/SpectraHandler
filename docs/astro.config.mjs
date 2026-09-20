// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://haeussma.github.io',
	base: '/SpectraHandler',
	integrations: [
		starlight({
			title: 'SpectraHandler',
			description: 'Spectral deconvolution for reaction data, on JAX and NumPyro.',
			lastUpdated: true,
			editLink: {
				baseUrl: 'https://github.com/haeussma/SpectraHandler/edit/main/docs/',
			},
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/haeussma/SpectraHandler',
				},
			],
			sidebar: [
				{
					label: 'Guides',
					items: [{ autogenerate: { directory: 'guides' } }],
				},
				{
					label: 'Reference',
					items: [{ autogenerate: { directory: 'reference' } }],
				},
			],
		}),
	],
});
