import os from "node:os";
import path from "node:path";

export function resolvePluginDataDirectory(
  environment: NodeJS.ProcessEnv = process.env,
  homeDirectory = os.homedir(),
  platform = process.platform,
): string {
  const explicit =
    environment.KONTEXT_PLUGIN_DATA ?? environment.PLUGIN_DATA ?? environment.CLAUDE_PLUGIN_DATA;
  if (explicit?.trim()) return path.resolve(explicit);
  if (platform === "win32") {
    const localAppData = environment.LOCALAPPDATA ?? environment.APPDATA;
    if (localAppData?.trim()) return path.resolve(localAppData, "kontext-brain");
  }
  const xdgDataHome = environment.XDG_DATA_HOME;
  if (xdgDataHome?.trim()) return path.resolve(xdgDataHome, "kontext-brain");
  return path.resolve(homeDirectory, ".local", "share", "kontext-brain");
}
