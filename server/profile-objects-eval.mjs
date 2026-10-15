/**
 * Profile image object cleanup evaluation (eval:profile-objects).
 *
 * A character's profile image is a file in the object store that no ImageAsset owns, so image_asset redaction never
 * reaches it. The API server deletes it on its own in two places, both checked here against the real server process
 * (spawned with a fresh data directory, a free port and a raised rate limit) and the files it leaves on disk:
 *
 *   - release on save: a save that stops naming a profile image deletes the file once no stored simulation of any
 *     owner names it — not a copy that shares the key, not an image job that can still run — and never a story image,
 *     a traversal-looking key, a file behind a link, or anything when the save is older than the stored version
 *   - startup sweep: a builder draft folder (objects/draft_sim_*) untouched for seven days loses every profile image no
 *     stored simulation names, in the default root and in an owner's personal-vault imageStoragePath; a recent or
 *     still-used draft, anything that is not a profile image, a junction, and every folder when the store cannot be
 *     read are left alone
 *
 * The age is set with utimes on files the API itself wrote, then the server is restarted, because the sweep runs once
 * before the server listens. No network access, GPU or keys are needed.
 *
 * Usage: node server/profile-objects-eval.mjs
 */
import { spawn } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { mkdir, mkdtemp, readdir, rm, symlink, utimes, writeFile } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createMockPng } from "./mock-comfyui.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const serverEntry = path.join(workspaceRoot, "server", "dynamicchat-server.mjs");

const PNG_DATA_URL = `data:image/png;base64,${createMockPng().toString("base64")}`;
const LOCAL_OWNER = "local_user";
const VAULT_OWNER = "vault_owner";
// Comfortably past the server's seven days, so the boundary itself is not what is being tested.
const ABANDONED_AT = new Date(Date.now() - 10 * 24 * 60 * 60 * 1000);
// Read by the server at startup. Inherited values from the developer's shell would change what is being tested.
const SERVER_ENV_KEYS = [
  "DYNAMICCHAT_CORS_ORIGIN",
  "DYNAMICCHAT_MAX_BODY_BYTES",
  "DYNAMICCHAT_RATE_LIMIT_WINDOW_MS",
  "DYNAMICCHAT_ALLOWED_HOSTS",
  "DYNAMICCHAT_DENIED_ORIGINS"
];

const checks = [];
const skipped = [];
const cleanups = [];

try {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "dynamicchat-profile-objects-"));
  cleanups.push(() => rm(tempRoot, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }));
  const dataDir = path.join(tempRoot, "data");
  const context = {
    dataDir,
    objectRoot: path.join(dataDir, "objects"),
    customRoot: path.join(tempRoot, "custom-root"),
    outsideDir: path.join(tempRoot, "outside")
  };
  await mkdir(context.outsideDir, { recursive: true });

  context.server = await startApiServer("first", dataDir);
  await runGroup("release on save", () => evaluateReleaseOnSave(context));
  await runGroup("shared keys", () => evaluateSharedKeys(context));
  await runGroup("unsafe keys", () => evaluateUnsafeKeys(context));
  await runGroup("live image jobs", () => evaluateLiveImageJobs(context));
  await runGroup("stale saves", () => evaluateStaleSaves(context));
  await runGroup("custom root", () => evaluateCustomRoot(context));
  await runGroup("links", () => evaluateLinksOnSave(context));
  await runGroup("odd shapes", () => evaluateOddShapes(context));
  await runGroup("startup sweep", () => evaluateStartupSweep(context));
  await runGroup("unreadable store", () => evaluateUnreadableStore(context));
} catch (error) {
  assertCheck("setup", false, `eval setup failed: ${error?.stack ?? error}`);
} finally {
  for (const cleanup of cleanups.reverse()) {
    await Promise.resolve()
      .then(cleanup)
      .catch((error) => console.error(`cleanup failed: ${error?.message ?? error}`));
  }
}

for (const check of checks) {
  const icon = check.level === "pass" ? "PASS" : "FAIL";
  console.log(`${icon} ${check.scope}: ${check.message}`);
}
for (const note of skipped) {
  console.log(`SKIP ${note}`);
}

const failures = checks.filter((check) => check.level === "fail");
console.log("");
console.log(`Profile object checks: ${checks.length - failures.length} passed, ${failures.length} failures`);

if (failures.length > 0) {
  process.exitCode = 1;
}

async function evaluateReleaseOnSave({ server, objectRoot }) {
  const scope = "release on save";
  const id = "sim_release";
  await persistObjects(server, id, ["profile_a1", "profile_a1_reference", "profile_p1", "profile_a2", "asset_story_1"]);

  await saveState(
    server,
    createState({
      id,
      updatedAt: at(1),
      profiles: {
        hero: {
          novelai: profileImage("novelai", `${id}/profile_a1.png`, `${id}/profile_a1_reference.png`),
          photoreal: profileImage("photoreal", `${id}/profile_p1.png`)
        },
        // The app never points a profile at a story image, which is exactly why the server must not take that on trust.
        rival: { photoreal: profileImage("photoreal", `${id}/asset_story_1.png`) }
      }
    })
  );
  assertCheck(
    scope,
    ["profile_a1.png", "profile_a1_reference.png", "profile_p1.png", "asset_story_1.png"].every((name) => objectExists(objectRoot, `${id}/${name}`)),
    "the first save of a simulation deletes nothing"
  );

  await saveState(server, createState({ id, updatedAt: at(2), profiles: { hero: { novelai: profileImage("novelai", `${id}/profile_a2.png`) }, rival: undefined } }));
  assertCheck(
    scope,
    !objectExists(objectRoot, `${id}/profile_a1.png`) && !objectExists(objectRoot, `${id}/profile_a1_reference.png`),
    "a replaced NovelAI profile is deleted together with its Precise Reference canvas",
    listObjects(objectRoot, id)
  );
  assertCheck(scope, !objectExists(objectRoot, `${id}/profile_p1.png`), "a cleared photoreal profile is deleted", listObjects(objectRoot, id));
  assertCheck(scope, objectExists(objectRoot, `${id}/profile_a2.png`), "the profile the save sets stays");
  assertCheck(scope, objectExists(objectRoot, `${id}/asset_story_1.png`), "a story image (not named profile_*) is never deleted, even when a released record names it");

  const served = await fetch(`${server.baseUrl}/objects/${id}/profile_a1.png`);
  await served.arrayBuffer();
  assertCheck(scope, served.status === 404, "GET /objects no longer serves the released face", `status=${served.status}`);
  assertCheck(
    scope,
    await waitForOutput(server, [`deleted ${id}/profile_a1.png`, `deleted ${id}/profile_a1_reference.png`]),
    "each deletion is logged by key",
    tail(server.output())
  );

  // Saving the same profiles again releases nothing.
  await saveState(server, createState({ id, updatedAt: at(3), profiles: { hero: { novelai: profileImage("novelai", `${id}/profile_a2.png`) } } }));
  assertCheck(scope, objectExists(objectRoot, `${id}/profile_a2.png`), "re-saving an unchanged profile keeps it");
}

async function evaluateSharedKeys({ server, objectRoot }) {
  const scope = "shared keys";
  const source = "sim_shared_source";
  const copy = "sim_shared_copy";
  await persistObjects(server, source, ["profile_c1", "profile_c2", "profile_c3", "profile_c4"]);
  await persistObjects(server, copy, ["profile_k1"]);

  // copySimulation carries the source's object keys into the copy instead of copying the files.
  await saveState(server, createState({ id: source, updatedAt: at(1), profiles: { hero: { photoreal: profileImage("photoreal", `${source}/profile_c1.png`) } } }));
  await saveState(server, createState({ id: copy, updatedAt: at(1), profiles: { hero: { photoreal: profileImage("photoreal", `${source}/profile_c1.png`) } } }));
  await saveState(server, createState({ id: source, updatedAt: at(2), profiles: { hero: { photoreal: profileImage("photoreal", `${source}/profile_c2.png`) } } }));
  assertCheck(scope, objectExists(objectRoot, `${source}/profile_c1.png`), "a profile the source replaced stays while a copy still uses the same key");

  await saveState(server, createState({ id: copy, updatedAt: at(2), profiles: { hero: { photoreal: profileImage("photoreal", `${copy}/profile_k1.png`) } } }));
  assertCheck(scope, !objectExists(objectRoot, `${source}/profile_c1.png`), "it is deleted once the copy lets go of it too", listObjects(objectRoot, source));

  // Owners are not a boundary for references: the check covers every stored simulation.
  await saveState(
    server,
    createState({ id: "sim_shared_other_owner", ownerId: "other_owner", updatedAt: at(1), profiles: { hero: { photoreal: profileImage("photoreal", `${source}/profile_c2.png`) } } })
  );
  await saveState(server, createState({ id: source, updatedAt: at(3), profiles: { hero: { photoreal: profileImage("photoreal", `${source}/profile_c3.png`) } } }));
  assertCheck(scope, objectExists(objectRoot, `${source}/profile_c2.png`), "a key another owner's simulation names stays");

  // The comparison is deliberately looser than the keys the server writes; a looser match can only keep a file.
  await saveState(
    server,
    createState({ id: "sim_shared_alias", updatedAt: at(1), profiles: { hero: { photoreal: profileImage("photoreal", `${source}\\profile_c3.png`) } } })
  );
  await saveState(server, createState({ id: source, updatedAt: at(4), profiles: { hero: { photoreal: profileImage("photoreal", `${source}/profile_c4.png`) } } }));
  assertCheck(scope, objectExists(objectRoot, `${source}/profile_c3.png`), "a reference spelled with a backslash still counts as one");
}

async function evaluateUnsafeKeys({ server, dataDir, objectRoot, outsideDir }) {
  const scope = "unsafe keys";
  const id = "sim_unsafe";
  // Bait: profile-named files a key could reach by traversal. sim_unsafe_victim's file is unreferenced, so only the
  // key's spelling stands between it and deletion.
  await persistObjects(server, "sim_unsafe_victim", ["profile_victim"]);
  await writeFile(path.join(dataDir, "profile_parent.png"), createMockPng());
  await writeFile(path.join(outsideDir, "profile_outside.png"), createMockPng());
  await mkdir(path.join(objectRoot, id, "nested"), { recursive: true });
  await writeFile(path.join(objectRoot, id, "nested", "profile_nested.png"), createMockPng());

  const unsafeKeys = [
    "../profile_parent.png",
    `${id}/../sim_unsafe_victim/profile_victim.png`,
    `${id}\\..\\sim_unsafe_victim\\profile_victim.png`,
    "sim_unsafe_victim/./profile_victim.png",
    "sim_unsafe_victim//profile_victim.png",
    `${id}/nested/profile_nested.png`,
    path.join(objectRoot, "sim_unsafe_victim", "profile_victim.png"),
    path.join(outsideDir, "profile_outside.png")
  ];
  await saveState(
    server,
    createState({
      id,
      updatedAt: at(1),
      profiles: Object.fromEntries(unsafeKeys.map((objectKey, index) => [`character_${index}`, { photoreal: profileImage("photoreal", objectKey) }]))
    })
  );
  await saveState(server, createState({ id, updatedAt: at(2), profiles: {} }));

  assertCheck(scope, objectExists(objectRoot, "sim_unsafe_victim/profile_victim.png"), "dot-dot, backslash, dot, empty-segment and absolute spellings delete nothing inside the root");
  assertCheck(scope, existsSync(path.join(dataDir, "profile_parent.png")), "a key that climbs out of the object root deletes nothing above it");
  assertCheck(scope, existsSync(path.join(outsideDir, "profile_outside.png")), "an absolute key outside the root deletes nothing");
  assertCheck(scope, objectExists(objectRoot, `${id}/nested/profile_nested.png`), "a key deeper than <folder>/<file> is never a profile pick");
}

async function evaluateLiveImageJobs({ server, objectRoot }) {
  const scope = "live image jobs";
  const id = "sim_jobs";
  await persistObjects(server, id, ["profile_e1", "profile_e2", "profile_n1", "profile_n1_reference", "profile_n2"]);
  const queuedJob = (status) => ({
    id: "imgjob_eval_photoreal",
    status,
    providerPayload: { provider: "photoreal", photorealReferences: [{ name: "char_1", characterId: "hero", objectKey: `${id}/profile_e1.png` }] }
  });
  const runJob = (status) => ({
    id: "imgjob_eval_novelai",
    status,
    providerPayload: { provider: "novelai", characterReference: { characterId: "hero", objectKey: `${id}/profile_n1_reference.png` } }
  });
  const withJobs = (state, photorealStatus, novelAiStatus) => ({
    ...state,
    imageJobs: [queuedJob(photorealStatus)],
    progressRuns: [{ id: "run_eval", simulationId: id, imageAssets: [], imageJobs: [runJob(novelAiStatus)] }]
  });

  await saveState(
    server,
    withJobs(
      createState({
        id,
        updatedAt: at(1),
        profiles: {
          hero: {
            photoreal: profileImage("photoreal", `${id}/profile_e1.png`),
            novelai: profileImage("novelai", `${id}/profile_n1.png`, `${id}/profile_n1_reference.png`)
          }
        }
      }),
      "queued",
      "generating"
    )
  );
  const replaced = createState({
    id,
    updatedAt: at(2),
    profiles: { hero: { photoreal: profileImage("photoreal", `${id}/profile_e2.png`), novelai: profileImage("novelai", `${id}/profile_n2.png`) } }
  });
  await saveState(server, withJobs(replaced, "queued", "generating"));
  assertCheck(scope, objectExists(objectRoot, `${id}/profile_e1.png`), "a replaced face a queued photoreal job will still load stays");
  assertCheck(scope, objectExists(objectRoot, `${id}/profile_n1_reference.png`), "a replaced reference canvas a running NovelAI job (in a progress run) names stays");
  assertCheck(scope, !objectExists(objectRoot, `${id}/profile_n1.png`), "the replaced profile no job names is deleted at once", listObjects(objectRoot, id));

  await saveState(server, withJobs({ ...replaced, simulation: { ...replaced.simulation, updatedAt: at(3) } }, "completed", "failed"));
  assertCheck(
    scope,
    !objectExists(objectRoot, `${id}/profile_e1.png`) && !objectExists(objectRoot, `${id}/profile_n1_reference.png`),
    "both are deleted once their jobs finish: a finished job's payload pins nothing",
    listObjects(objectRoot, id)
  );
  assertCheck(scope, objectExists(objectRoot, `${id}/profile_e2.png`) && objectExists(objectRoot, `${id}/profile_n2.png`), "the current profiles stay");
}

async function evaluateStaleSaves({ server, objectRoot }) {
  const scope = "stale saves";
  const id = "sim_stale";
  await persistObjects(server, id, ["profile_s1", "profile_s2"]);
  await saveState(server, createState({ id, updatedAt: at(1), profiles: { hero: { photoreal: profileImage("photoreal", `${id}/profile_s1.png`) } } }));
  await saveState(server, createState({ id, updatedAt: at(2), profiles: { hero: { photoreal: profileImage("photoreal", `${id}/profile_s2.png`) } } }));
  assertCheck(scope, !objectExists(objectRoot, `${id}/profile_s1.png`), "a newer save releases the profile it replaced");

  // The app fires saves without waiting for the previous one, so an older snapshot can land after a newer one.
  await saveState(server, createState({ id, updatedAt: at(1), profiles: { hero: { photoreal: profileImage("photoreal", `${id}/profile_s1.png`) } } }));
  assertCheck(scope, objectExists(objectRoot, `${id}/profile_s2.png`), "an older snapshot landing late does not delete the profile the newer save set");
  assertCheck(scope, await waitForOutput(server, [`this save of ${id} is older than the stored version`]), "the kept release is logged", tail(server.output()));

  const unknown = "sim_stale_undated";
  await persistObjects(server, unknown, ["profile_u1", "profile_u2"]);
  await saveState(server, createState({ id: unknown, updatedAt: "not a date", profiles: { hero: { photoreal: profileImage("photoreal", `${unknown}/profile_u1.png`) } } }));
  await saveState(server, createState({ id: unknown, updatedAt: at(2), profiles: { hero: { photoreal: profileImage("photoreal", `${unknown}/profile_u2.png`) } } }));
  assertCheck(scope, objectExists(objectRoot, `${unknown}/profile_u1.png`), "a save whose order against the stored version cannot be read releases nothing");
}

async function evaluateCustomRoot({ server, objectRoot, customRoot }) {
  const scope = "custom root";
  const vault = await api(server, "/personal-api-vault", { method: "PUT", owner: VAULT_OWNER, body: { imageStoragePath: customRoot } });
  assertCheck(scope, vault.status === 204, "an owner can point its images at its own folder", describe(vault));

  const id = "sim_vault";
  await persistObjects(server, id, ["profile_v1", "profile_v2"], VAULT_OWNER);
  assertCheck(
    scope,
    objectExists(customRoot, `${id}/profile_v1.png`) && !objectExists(objectRoot, `${id}/profile_v1.png`),
    "the picks were written to the owner's folder, not the default root"
  );
  await saveState(server, createState({ id, ownerId: VAULT_OWNER, updatedAt: at(1), profiles: { hero: { photoreal: profileImage("photoreal", `${id}/profile_v1.png`) } } }), VAULT_OWNER);
  await saveState(server, createState({ id, ownerId: VAULT_OWNER, updatedAt: at(2), profiles: { hero: { photoreal: profileImage("photoreal", `${id}/profile_v2.png`) } } }), VAULT_OWNER);
  assertCheck(scope, !objectExists(customRoot, `${id}/profile_v1.png`), "a released profile is deleted from the owner's folder", listObjects(customRoot, id));
  assertCheck(scope, objectExists(customRoot, `${id}/profile_v2.png`), "the owner's current profile stays");
}

async function evaluateLinksOnSave({ server, objectRoot, outsideDir }) {
  const scope = "links";
  const target = path.join(outsideDir, "linked");
  await mkdir(target, { recursive: true });
  await writeFile(path.join(target, "profile_bait.png"), createMockPng());
  if (!(await createJunction(target, path.join(objectRoot, "sim_linked_folder"), scope))) {
    return;
  }

  // On Windows a path THROUGH a junction reads as an ordinary file, so a check of the file alone would follow it.
  const id = "sim_links";
  await saveState(server, createState({ id, updatedAt: at(1), profiles: { hero: { photoreal: profileImage("photoreal", "sim_linked_folder/profile_bait.png") } } }));
  await saveState(server, createState({ id, updatedAt: at(2), profiles: {} }));
  assertCheck(scope, existsSync(path.join(target, "profile_bait.png")), "a released key whose folder is a junction does not delete the file it points at");
}

async function evaluateOddShapes({ server }) {
  const scope = "odd shapes";
  const id = "sim_odd";
  const odd = {
    ...createState({ id, updatedAt: at(1) }),
    visualProfiles: [null, "profile", { characterId: "a", profileImages: [] }, { characterId: "b", profileImages: { novelai: null, photoreal: { objectKey: 42, referenceObjectKey: {} } } }],
    imageJobs: "not jobs",
    progressRuns: [{ id: "run_odd", imageJobs: { not: "an array" } }]
  };
  const first = await api(server, `/simulations/${id}/state`, { method: "PUT", body: odd });
  const second = await api(server, `/simulations/${id}/state`, { method: "PUT", body: createState({ id, updatedAt: at(2) }) });
  assertCheck(scope, first.status === 204 && second.status === 204, "malformed profile records and job lists neither fail a save nor its cleanup", `${describe(first)} / ${describe(second)}`);
}

async function evaluateStartupSweep(context) {
  const scope = "startup sweep";
  const { objectRoot, customRoot, outsideDir } = context;
  let server = context.server;

  // Everything below is written by the API itself, as the builder would have written it.
  await persistObjects(server, "draft_sim_abandon_1", ["profile_x", "profile_x_reference"]);
  await persistObjects(server, "draft_sim_created_2", ["profile_used", "profile_superseded"]);
  await saveState(
    server,
    createState({ id: "sim_from_draft", updatedAt: at(1), profiles: { hero: { photoreal: profileImage("photoreal", "draft_sim_created_2/profile_used.png") } } })
  );
  await persistObjects(server, "draft_sim_recent_3", ["profile_recent"]);
  await persistObjects(server, "draft_sim_stray_4", ["profile_y"]);
  await writeFile(path.join(objectRoot, "draft_sim_stray_4", "notes.txt"), "not a profile image");
  await persistObjects(server, "draft_sim_vault_5", ["profile_z"], VAULT_OWNER);
  await persistObjects(server, "draft_sim_jobonly_6", ["profile_job"]);
  await saveState(server, {
    ...createState({ id: "sim_job_holder", updatedAt: at(1) }),
    imageJobs: [
      {
        id: "imgjob_eval_draft",
        status: "queued",
        providerPayload: { provider: "photoreal", photorealReferences: [{ name: "char_1", objectKey: "draft_sim_jobonly_6/profile_job.png" }] }
      }
    ]
  });
  await persistObjects(server, "draft_simulation", ["profile_q"]);
  await persistObjects(server, "sim_orphan", ["profile_orphan"]);
  await mkdir(path.join(objectRoot, "draft_sim_empty_8"), { recursive: true });
  const junctionTarget = path.join(outsideDir, "draft-target");
  await mkdir(junctionTarget, { recursive: true });
  await writeFile(path.join(junctionTarget, "profile_bait.png"), createMockPng());
  const junctionCreated = await createJunction(junctionTarget, path.join(objectRoot, "draft_sim_junction_7"), scope);

  await server.stop();
  for (const folder of ["draft_sim_abandon_1", "draft_sim_created_2", "draft_sim_stray_4", "draft_sim_jobonly_6", "draft_simulation", "sim_orphan", "draft_sim_empty_8"]) {
    await ageTree(path.join(objectRoot, folder));
  }
  await ageTree(path.join(customRoot, "draft_sim_vault_5"));
  await ageTree(junctionTarget);

  server = await startApiServer("second", context.dataDir);
  context.server = server;

  assertCheck(scope, !existsSync(path.join(objectRoot, "draft_sim_abandon_1")), "an abandoned draft folder goes entirely, reference canvas included", listObjects(objectRoot, "draft_sim_abandon_1"));
  assertCheck(scope, objectExists(objectRoot, "draft_sim_created_2/profile_used.png"), "a draft that became a simulation keeps the pick the simulation uses");
  assertCheck(scope, !objectExists(objectRoot, "draft_sim_created_2/profile_superseded.png"), "...and loses the pick replaced before 생성, which no save could ever release");
  assertCheck(scope, objectExists(objectRoot, "draft_sim_recent_3/profile_recent.png"), "a draft touched within seven days is left alone (its builder may still be open)");
  assertCheck(
    scope,
    !objectExists(objectRoot, "draft_sim_stray_4/profile_y.png") && objectExists(objectRoot, "draft_sim_stray_4/notes.txt"),
    "a file that is not a profile image survives the sweep and keeps its folder",
    listObjects(objectRoot, "draft_sim_stray_4")
  );
  assertCheck(scope, !existsSync(path.join(customRoot, "draft_sim_vault_5")), "an abandoned draft in an owner's imageStoragePath is swept too", listObjects(customRoot, "draft_sim_vault_5"));
  assertCheck(scope, objectExists(objectRoot, "draft_sim_jobonly_6/profile_job.png"), "a draft pick a queued job still names stays");
  assertCheck(scope, objectExists(objectRoot, "draft_simulation/profile_q.png"), "a folder not named like createId(\"draft_sim\") is not a draft");
  assertCheck(scope, objectExists(objectRoot, "sim_orphan/profile_orphan.png"), "a simulation's own folder is not swept, even for a profile nothing names");
  assertCheck(scope, !existsSync(path.join(objectRoot, "draft_sim_empty_8")), "an old empty draft folder is removed");
  if (junctionCreated) {
    assertCheck(scope, existsSync(path.join(junctionTarget, "profile_bait.png")), "a junction named like a draft is not followed");
  }
  assertCheck(
    scope,
    await waitForOutput(server, ["removed the abandoned draft folder draft_sim_abandon_1", "draft_sim_created_2/profile_superseded.png"]),
    "the sweep logs what it removed",
    tail(server.output())
  );

  const health = await api(server, "/health");
  assertCheck(scope, health.status === 200, "the server answers after the sweep", describe(health));
  const stored = await api(server, "/simulations/sim_from_draft");
  assertCheck(
    scope,
    stored.status === 200 && stored.json?.visualProfiles?.[0]?.profileImages?.photoreal?.objectKey === "draft_sim_created_2/profile_used.png",
    "the stored simulation still names its draft-folder profile",
    describe(stored)
  );
}

async function evaluateUnreadableStore(context) {
  const scope = "unreadable store";
  const { objectRoot, dataDir } = context;
  await context.server.stop();

  const folder = path.join(objectRoot, "draft_sim_unread_9");
  await mkdir(folder, { recursive: true });
  await writeFile(path.join(folder, "profile_w.png"), createMockPng());
  await ageTree(folder);

  // Two ways a store names no references: it does not parse, or it parses without a simulations map.
  for (const [label, contents] of [
    ["not JSON", "{ this is not json"],
    ["no simulations map", JSON.stringify({ simulations: [] })]
  ]) {
    await writeFile(path.join(dataDir, "state.json"), contents);
    const server = await startApiServer(`store-${label.replace(/\W+/gu, "-")}`, dataDir);
    context.server = server;
    assertCheck(scope, existsSync(path.join(folder, "profile_w.png")), `a store with ${label} deletes nothing at startup`);
    assertCheck(scope, await waitForOutput(server, ["Profile image cleanup at startup stopped"]), `the skipped sweep is logged (${label})`, tail(server.output()));
    await server.stop();
  }

  // The control: the same folder does go once the store reads, so the two checks above are not passing by default.
  await writeFile(path.join(dataDir, "state.json"), `${JSON.stringify({ simulations: {} })}\n`);
  const server = await startApiServer("store-readable", dataDir);
  context.server = server;
  assertCheck(scope, !existsSync(folder), "with a readable store the same folder is swept", listObjects(objectRoot, "draft_sim_unread_9"));
  await server.stop();
}

function createState({ id, ownerId = LOCAL_OWNER, updatedAt, profiles = {} }) {
  return {
    simulation: {
      id,
      ownerId,
      title: id,
      description: "",
      promptMode: "balanced",
      contentRating: "general",
      activeSessionId: "session_eval",
      realtimeImageEnabled: false,
      defaultChatModelProfile: "",
      createdAt: at(0),
      updatedAt
    },
    visualProfiles: Object.entries(profiles).map(([characterId, profileImages]) => ({
      id: `visual_${characterId}`,
      simulationId: id,
      characterId,
      displayName: characterId,
      positivePrompt: "",
      negativePrompt: "",
      referenceImageAssetIds: [],
      defaultSafetyLevel: "safe",
      ...(profileImages ? { profileImages } : {})
    })),
    messages: [],
    imageAssets: [],
    imageJobs: [],
    progressRuns: [],
    auditLog: []
  };
}

function profileImage(backend, objectKey, referenceObjectKey) {
  return {
    backend,
    assetId: path.posix.basename(objectKey.replace(/\\/gu, "/")).replace(/\.png$/u, ""),
    objectKey,
    ...(referenceObjectKey ? { referenceObjectKey } : {}),
    prompt: "eval",
    selectedAt: at(0)
  };
}

// Distinct, ordered save times; only their order matters to the server.
function at(minute) {
  return new Date(Date.UTC(2026, 9, 1, 0, minute)).toISOString();
}

async function persistObjects(server, simulationId, assetIds, owner) {
  const result = await api(server, `/simulations/${simulationId}/assets`, {
    method: "POST",
    owner,
    body: { assets: assetIds.map((id) => ({ id, simulationId, dataUrl: PNG_DATA_URL, mimeType: "image/png" })) }
  });
  const objectKeys = assetIds.map((id) => `${simulationId}/${id}.png`);
  const stored = Array.isArray(result.json) && objectKeys.every((objectKey) => result.json.some((asset) => asset?.objectKey === objectKey));
  if (result.status !== 200 || !stored) {
    throw new Error(`POST /simulations/${simulationId}/assets did not store ${objectKeys.join(", ")}: ${describe(result)}`);
  }
  return objectKeys;
}

async function saveState(server, state, owner) {
  const result = await api(server, `/simulations/${state.simulation.id}/state`, { method: "PUT", owner, body: state });
  if (result.status !== 204) {
    throw new Error(`PUT /simulations/${state.simulation.id}/state answered ${describe(result)}`);
  }
}

// Directory junctions need no privilege on Windows (symlinks do); elsewhere the type is ignored and a plain directory
// symlink is made, which needs none either.
async function createJunction(target, linkPath, scope) {
  try {
    await symlink(target, linkPath, "junction");
    return true;
  } catch (error) {
    skipped.push(`${scope}: could not create a junction (${error?.code ?? error}), so the link checks did not run`);
    return false;
  }
}

// Backdates a folder and everything directly in it, the way seven idle days would leave them. The folder goes last:
// writing its files moved its own time stamp.
async function ageTree(folder) {
  for (const name of await readdir(folder)) {
    await utimes(path.join(folder, name), ABANDONED_AT, ABANDONED_AT);
  }
  await utimes(folder, ABANDONED_AT, ABANDONED_AT);
}

function objectExists(root, objectKey) {
  return existsSync(path.join(root, ...objectKey.split("/")));
}

// For failure messages: what is actually in the folder the check was about.
function listObjects(root, folder) {
  const directory = path.join(root, folder);
  if (!existsSync(directory)) {
    return `${folder}/ is gone`;
  }
  try {
    return `${folder}/ holds: ${readdirSync(directory).join(", ") || "(nothing)"}`;
  } catch (error) {
    return `${folder}/ is unreadable (${error?.code ?? error})`;
  }
}

async function startApiServer(label, dataDir) {
  await mkdir(dataDir, { recursive: true });
  const port = await findFreePort();
  const env = { ...process.env };
  for (const key of SERVER_ENV_KEYS) {
    delete env[key];
  }
  Object.assign(env, { DYNAMICCHAT_DATA_DIR: dataDir, DYNAMICCHAT_API_PORT: String(port), DYNAMICCHAT_RATE_LIMIT_MAX: "10000" });
  const child = spawn(process.execPath, [serverEntry], { cwd: workspaceRoot, env, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  let output = "";
  child.stdout.on("data", (chunk) => {
    output += chunk;
  });
  child.stderr.on("data", (chunk) => {
    output += chunk;
  });
  const exited = new Promise((resolve) => child.once("exit", resolve));
  const server = {
    label,
    port,
    baseUrl: `http://127.0.0.1:${port}`,
    output: () => output,
    async stop() {
      if (child.exitCode === null && child.signalCode === null) {
        child.kill();
        await Promise.race([exited, delay(5_000)]);
      }
    }
  };
  cleanups.push(() => server.stop());
  const startedAt = Date.now();
  while (Date.now() - startedAt < 20_000) {
    if (child.exitCode !== null) {
      throw new Error(`${label} server exited during startup (code ${child.exitCode}):\n${output}`);
    }
    try {
      const health = await fetch(`${server.baseUrl}/health`, { signal: AbortSignal.timeout(1_000) });
      await health.text();
      if (health.ok) {
        return server;
      }
    } catch {
      // Not listening yet.
    }
    await delay(100);
  }
  await server.stop();
  throw new Error(`${label} server did not answer /health within 20s:\n${output}`);
}

async function api(server, pathname, { method = "GET", body, owner } = {}) {
  const headers = {};
  if (owner) {
    headers["x-dynamicchat-owner-id"] = owner;
  }
  let payload;
  if (body !== undefined) {
    headers["content-type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const response = await fetch(`${server.baseUrl}${pathname}`, { method, headers, body: payload });
  const text = await response.text();
  let json;
  try {
    json = text ? JSON.parse(text) : undefined;
  } catch {
    json = undefined;
  }
  return { status: response.status, json, text };
}

function findFreePort() {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.unref();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close(() => resolve(port));
    });
  });
}

// The server logs before it answers, but its stdout reaches us through a pipe on its own schedule.
async function waitForOutput(server, needles, timeoutMs = 3_000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (needles.every((needle) => server.output().includes(needle))) {
      return true;
    }
    await delay(25);
  }
  return needles.every((needle) => server.output().includes(needle));
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function describe(result) {
  return `status=${result?.status} body=${String(result?.text ?? "").slice(0, 300)}`;
}

function tail(text) {
  return `server log: ${String(text).slice(-600)}`;
}

async function runGroup(scope, evaluate) {
  try {
    await evaluate();
  } catch (error) {
    assertCheck(scope, false, `threw before finishing: ${error?.stack ?? error}`);
  }
}

function assertCheck(scope, passed, message, detail) {
  checks.push({
    level: passed ? "pass" : "fail",
    scope,
    message: passed || detail === undefined ? message : `${message} — ${detail}`
  });
}
