import { KontextLoader } from "@kontext-brain/loader";
import { KontextToolServer } from "./kontext-tool-server.js";

async function main(): Promise<void> {
  const configPath = process.argv[2];
  if (!configPath) {
    process.stderr.write("Usage: kontext-tool-server <config.yaml>\n");
    process.stderr.write("  Starts an MCP tool server over stdio.\n\n");
    process.stderr.write("Claude Desktop config example:\n");
    process.stderr.write(
      `  {
    "mcpServers": {
      "kontext": {
        "command": "kontext-tool-server",
        "args": ["/path/to/kontext.yaml"]
      }
    }
  }\n`,
    );
    process.exit(1);
  }

  const agent = await KontextLoader.fromFile(configPath);
  await new KontextToolServer(agent).start();
}

main().catch((err) => {
  process.stderr.write(`kontext-tool-server error: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
