/**
 * Fail-closed ACL SQL predicate shared by every read surface in this package.
 * Expects `$2 = principal.subjectId` and `$3 = principal.groupIds` in the query.
 */
export function aclPredicate(alias: string): string {
  return `(COALESCE((${alias}.acl->>'organizationWide')::boolean, false)
    OR COALESCE(${alias}.acl->'subjectIds', '[]'::jsonb) ? $2
    OR EXISTS (
      SELECT 1 FROM jsonb_array_elements_text(COALESCE(${alias}.acl->'groupIds', '[]'::jsonb)) g(id)
      WHERE g.id = ANY($3::text[])
    ))`;
}

export function toIsoString(value: unknown): string {
  return value instanceof Date ? value.toISOString() : new Date(String(value)).toISOString();
}

export function toConnectionError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}
