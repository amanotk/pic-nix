// graphify OpenCode plugin
// Injects a knowledge graph reminder before bash tool calls when the graph exists.
import { existsSync } from "fs";
import { join } from "path";

export const GraphifyPlugin = async ({ directory }) => {
  let reminded = false;
  const graphPath = join(directory, "docs", "graphify", "graph.json");

  return {
    "tool.execute.before": async (input, output) => {
      if (reminded) return;
      if (!existsSync(graphPath)) return;

      if (input.tool === "bash") {
        output.args.command =
          'echo "[graphify] Knowledge graph available at docs/graphify/. Read docs/graphify/GRAPH_REPORT.md before architecture searches. Use graphify query/path/explain with --graph docs/graphify/graph.json." && ' +
          output.args.command;
        reminded = true;
      }
    },
  };
};
