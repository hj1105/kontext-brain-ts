import { z } from "zod";

const id = z.string().min(1).max(4096);
export const kontextSessionSourceRequestSchema = z.object({ sessionId: id });
const session = z.object({ sessionId: id, workspaceId: id, agent: z.enum(["codex", "claude"]) });
export const kontextSessionSourceListSchema = z
  .object({
    runtimeId: id,
    scope: z.literal("readable_native_sessions"),
    registrationVersion: z.literal(1).optional(),
    sessions: z.array(session).max(1000),
  })
  .refine(
    (value) =>
      new Set(value.sessions.map((entry) => entry.sessionId)).size === value.sessions.length,
  );

export const kontextSessionSourcePayloadSchema = z.object({
  schemaVersion: z.literal(1),
  origin: z.object({
    kind: z.literal("kondex_session"),
    runtimeId: id,
    executionHostId: id,
    workspaceId: id,
    workspaceKind: z.enum(["git-worktree", "folder"]),
    wslDistro: id.nullable(),
    sessionId: id,
    provider: z.enum(["codex", "claude"]),
  }),
  journalCursor: z.object({ epoch: id, sequence: z.number().int().nonnegative() }),
  scope: z.literal("journal_user_assistant_text"),
  messages: z
    .array(
      z.object({
        itemId: id,
        revision: z.number().int().positive(),
        sequence: z.number().int().positive(),
        observedAt: z.number().finite(),
        recovered: z.boolean(),
        role: z.enum(["user", "assistant"]),
        blocks: z
          .array(z.object({ index: z.number().int().nonnegative(), text: z.string().min(1) }))
          .min(1)
          .max(256),
      }),
    )
    .min(1)
    .max(256),
  excluded: z.object({
    items: z.number().int().nonnegative(),
    blocks: z.number().int().nonnegative(),
    unconfirmedSubmissions: z.number().int().nonnegative(),
  }),
  providerSharing: z.literal("not_granted"),
  normativeApproval: z.literal("not_granted"),
});
export const kontextSessionSourcePreviewSchema = kontextSessionSourcePayloadSchema
  .extend({
    contentDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
    registration: z.literal("not_registered"),
  })
  .refine(
    (value) =>
      new Set(value.messages.map((message) => message.itemId)).size === value.messages.length &&
      value.messages.every(
        (message, index) =>
          message.sequence <= value.journalCursor.sequence &&
          (index === 0 ||
            message.sequence >=
              (value.messages[index - 1]?.sequence ?? Number.POSITIVE_INFINITY)) &&
          message.blocks.every(
            (block, blockIndex) =>
              blockIndex === 0 ||
              block.index > (message.blocks[blockIndex - 1]?.index ?? Number.POSITIVE_INFINITY),
          ),
      ),
    "Session source provenance is inconsistent",
  );
export type KontextSessionSourcePreview = z.infer<typeof kontextSessionSourcePreviewSchema>;
