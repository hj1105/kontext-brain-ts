export interface ExtractionJobKey {
  readonly organizationId: string;
  readonly resourceId: string;
  readonly contentHash: string;
  readonly ontologyHash: string;
}

export interface ExtractionJob extends ExtractionJobKey {
  readonly state: "pending" | "running" | "succeeded" | "failed" | "dead";
  readonly attempts: number;
  readonly availableAt: string;
  readonly lockedBy?: string;
  readonly lockedUntil?: string;
  readonly lastError?: string;
}

export interface ExtractionJobQueue {
  enqueue(key: ExtractionJobKey): Promise<boolean>;
  claim(
    organizationId: string,
    workerId: string,
    limit: number,
    leaseMs: number,
  ): Promise<readonly ExtractionJob[]>;
  succeed(key: ExtractionJobKey, workerId: string): Promise<void>;
  fail(
    key: ExtractionJobKey,
    workerId: string,
    error: string,
    retryAt: Date | null,
    maxAttempts: number,
  ): Promise<void>;
}
