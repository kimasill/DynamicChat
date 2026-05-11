import { createId } from "../lib/id";
import type {
  AppState,
  AuditAction,
  AuditEvent,
  AuditResourceType,
  RedactionRequest,
  RedactionTargetType,
  SecuritySettings,
  Simulation,
  TenantScope
} from "../types";

export function createTenantScope(simulation: Pick<Simulation, "id" | "ownerId">): TenantScope {
  return {
    ownerId: simulation.ownerId,
    workspaceId: "local_workspace",
    projectId: simulation.id,
    environment: "local"
  };
}

export function createSecuritySettings(simulation: Pick<Simulation, "id" | "ownerId">): SecuritySettings {
  return {
    localUserId: simulation.ownerId,
    scope: createTenantScope(simulation),
    secretStorage: "server_dev_store",
    browserSecretCacheEnabled: false,
    warning: "개발 서버는 API 키를 .dynamicchat-data/dev-secrets.json에 저장합니다. 프로덕션에서는 외부 vault/envelope encryption으로 교체해야 합니다."
  };
}

export function normalizeSecuritySettings(
  simulation: Pick<Simulation, "id" | "ownerId">,
  security?: Partial<SecuritySettings>
): SecuritySettings {
  const fallback = createSecuritySettings(simulation);
  const scope = {
    ...fallback.scope,
    ...security?.scope,
    ownerId: simulation.ownerId,
    projectId: simulation.id
  };

  return {
    ...fallback,
    ...security,
    localUserId: security?.localUserId ?? simulation.ownerId,
    scope,
    browserSecretCacheEnabled: false
  };
}

export function createAuditEvent(
  state: AppState,
  action: AuditAction,
  resourceType: AuditResourceType,
  resourceId: string,
  metadata: Record<string, unknown> = {}
): AuditEvent {
  return {
    id: createId("audit"),
    simulationId: state.simulation.id,
    ownerId: state.simulation.ownerId,
    scope: state.security.scope,
    action,
    resourceType,
    resourceId,
    metadata,
    createdAt: new Date().toISOString()
  };
}

export function createRedactionRequest(
  state: AppState,
  targetType: RedactionTargetType,
  targetId: string,
  reason: string,
  neuralMapNodeIds: string[] = []
): RedactionRequest {
  const now = new Date().toISOString();
  return {
    id: createId("redact"),
    simulationId: state.simulation.id,
    ownerId: state.simulation.ownerId,
    scope: state.security.scope,
    targetType,
    targetId,
    reason,
    neuralMapNodeIds,
    status: "applied",
    createdAt: now,
    completedAt: now
  };
}

export function createScopeHeaders(state?: AppState): Record<string, string> {
  const scope = state?.security?.scope ?? createTenantScope(state?.simulation ?? { id: "local_project", ownerId: "local_user" });
  return {
    "x-dynamicchat-owner-id": scope.ownerId,
    "x-dynamicchat-workspace-id": scope.workspaceId,
    "x-dynamicchat-project-id": scope.projectId,
    "x-dynamicchat-environment": scope.environment
  };
}

export function createScopeMetadata(state: AppState): Record<string, string> {
  const scope = state.security.scope;
  return {
    owner_id: scope.ownerId,
    tenant_id: scope.ownerId,
    workspace_id: scope.workspaceId,
    project_id: scope.projectId,
    environment: scope.environment
  };
}
