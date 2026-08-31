const subscriptionEnvironmentNames = [
  "PATH",
  "HOME",
  "USERPROFILE",
  "LOCALAPPDATA",
  "APPDATA",
  "XDG_CONFIG_HOME",
  "XDG_DATA_HOME",
  "XDG_CACHE_HOME",
  "CODEX_HOME",
  "CLAUDE_CONFIG_DIR",
  "TMPDIR",
  "TEMP",
  "TMP",
  "LANG",
  "LC_ALL",
  "SHELL",
  "SYSTEMROOT",
  "COMSPEC",
  "PATHEXT",
] as const;

export function subscriptionRuntimeEnvironment(
  dataDirectory: string,
  source: NodeJS.ProcessEnv = process.env,
): Record<string, string> {
  const environment: Record<string, string> = { KONTEXT_PLUGIN_DATA: dataDirectory };
  for (const name of subscriptionEnvironmentNames) {
    const value = source[name];
    if (value !== undefined) environment[name] = value;
  }
  return environment;
}
