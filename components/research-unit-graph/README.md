# Research unit graph

This local Streamlit v2 component renders the Research Harness unit map with a
bundled, pinned Cytoscape.js dependency. It is installed by the repository's
`requirements.txt`; the prebuilt frontend is included so normal installation
does not require Node.js or a browser-side CDN.

To rebuild after editing the TypeScript frontend:

```sh
cd components/research-unit-graph/research_unit_graph/frontend
npm ci
npm run build
```

The graph is a read-only view of task and unit relationships. Its only event
returns a selected unit ID to the existing inspector.
