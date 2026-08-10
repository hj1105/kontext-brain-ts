import type { AccessControlList, Principal } from "./domain.js";

export interface AccessPolicy {
  canAccess(principal: Principal, acl: AccessControlList): boolean;
}

/** Fail-closed ACL evaluation shared by retrieval and graph adapters. */
export class DefaultAccessPolicy implements AccessPolicy {
  canAccess(principal: Principal, acl: AccessControlList): boolean {
    if (acl.organizationWide === true) return true;
    if (acl.subjectIds?.includes(principal.subjectId)) return true;
    const principalGroups = new Set(principal.groupIds);
    if (acl.groupIds?.some((groupId) => principalGroups.has(groupId))) return true;
    return false;
  }
}
