-- DynamicChat MVP persistence schema.
-- Target database: PostgreSQL 15+.

create table if not exists users (
  id text primary key,
  display_name text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists simulations (
  id text primary key,
  owner_id text not null references users(id),
  workspace_id text not null default 'local_workspace',
  project_id text not null,
  environment text not null default 'local',
  title text not null,
  description text not null default '',
  content_rating text not null default 'general',
  active_session_id text not null,
  default_chat_model_profile text not null,
  realtime_image_enabled boolean not null default true,
  created_at timestamptz not null,
  updated_at timestamptz not null
);

create table if not exists prompt_modules (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  parent_id text references prompt_modules(id) on delete set null,
  kind text not null,
  title text not null,
  body text not null,
  enabled boolean not null default true,
  priority integer not null default 50,
  activation_tags text[] not null default '{}',
  character_id text,
  token_policy text not null,
  version integer not null default 1,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null
);

create table if not exists characters (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  name text not null,
  role text not null default '',
  summary text not null default '',
  relationship text not null default '',
  current_mood text not null default ''
);

alter table prompt_modules
  add constraint prompt_modules_character_fk
  foreign key (character_id) references characters(id) on delete set null;

create table if not exists character_visual_profiles (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  character_id text not null references characters(id) on delete cascade,
  display_name text not null,
  positive_prompt text not null,
  negative_prompt text not null default '',
  default_outfit_prompt text not null default '',
  outfit_prompts jsonb not null default '{}'::jsonb,
  expression_prompts jsonb not null default '{}'::jsonb,
  reference_image_asset_ids text[] not null default '{}',
  default_safety_level text not null
);

create table if not exists image_generation_profiles (
  id text primary key,
  simulation_id text not null unique references simulations(id) on delete cascade,
  enabled boolean not null default true,
  provider text not null default 'novelai',
  model text not null,
  width integer not null,
  height integer not null,
  steps integer not null,
  prompt_guidance numeric not null,
  count_min integer not null,
  count_max integer not null,
  quality_prompt text not null default '',
  style_prompt text not null default '',
  artist_prompt text not null default '',
  negative_prompt text not null default '',
  safety_level text not null,
  user_rules text not null default '',
  trigger_mode text not null,
  cooldown_turns integer not null default 0
);

create table if not exists sessions (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  started_at timestamptz not null default now(),
  ended_at timestamptz,
  handoff_pack_id text,
  continuity_status text not null default 'unchecked'
);

create table if not exists chat_messages (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  session_id text not null references sessions(id) on delete cascade,
  role text not null,
  content text not null,
  referenced_node_ids text[] not null default '{}',
  image_asset_ids text[] not null default '{}',
  created_at timestamptz not null
);

create table if not exists memory_events (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  session_id text not null references sessions(id) on delete cascade,
  actor_id text references characters(id) on delete set null,
  actor_name text,
  content text not null,
  importance numeric not null,
  tags text[] not null default '{}',
  source_turn_id text references chat_messages(id) on delete set null,
  neural_map_node_id text,
  created_at timestamptz not null
);

create table if not exists context_packs (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  session_id text not null references sessions(id) on delete cascade,
  objective text not null,
  token_budget integer not null,
  evidence jsonb not null default '[]'::jsonb,
  decisions jsonb not null default '[]'::jsonb,
  blockers jsonb not null default '[]'::jsonb,
  source text not null,
  created_at timestamptz not null
);

create table if not exists session_handoffs (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  previous_session_id text not null references sessions(id) on delete cascade,
  next_session_id text not null references sessions(id) on delete cascade,
  summary text not null,
  evidence_node_ids text[] not null default '{}',
  source text not null,
  error text,
  created_at timestamptz not null
);

create table if not exists continuity_checks (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  previous_session_id text not null references sessions(id) on delete cascade,
  next_session_id text not null references sessions(id) on delete cascade,
  handoff_id text not null references session_handoffs(id) on delete cascade,
  status text not null,
  facts jsonb not null default '[]'::jsonb,
  warnings jsonb not null default '[]'::jsonb,
  checked_at timestamptz not null
);

create table if not exists prompt_module_usages (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  session_id text not null references sessions(id) on delete cascade,
  turn_id text not null references chat_messages(id) on delete cascade,
  module_id text not null references prompt_modules(id) on delete cascade,
  module_title text not null,
  token_policy text not null,
  source text not null,
  reason text not null,
  score numeric not null,
  created_at timestamptz not null
);

create table if not exists sidecar_traces (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  session_id text not null references sessions(id) on delete cascade,
  turn_id text not null references chat_messages(id) on delete cascade,
  source text not null,
  status text not null,
  errors jsonb not null default '[]'::jsonb,
  raw_preview text,
  created_at timestamptz not null
);

create table if not exists image_jobs (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  session_id text not null references sessions(id) on delete cascade,
  turn_id text not null references chat_messages(id) on delete cascade,
  status text not null,
  reason text not null,
  prompt text not null,
  negative_prompt text not null,
  provider_payload jsonb not null default '{}'::jsonb,
  asset_ids text[] not null default '{}',
  context_node_ids text[] not null default '{}',
  error text,
  policy_warnings jsonb not null default '[]'::jsonb,
  representative_asset_id text,
  created_at timestamptz not null,
  updated_at timestamptz,
  completed_at timestamptz
);

create table if not exists turn_traces (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  session_id text not null references sessions(id) on delete cascade,
  turn_id text not null references chat_messages(id) on delete cascade,
  user_message_id text not null references chat_messages(id) on delete cascade,
  assistant_message_id text not null references chat_messages(id) on delete cascade,
  context_pack_id text not null references context_packs(id) on delete cascade,
  prompt_module_usage_ids text[] not null default '{}',
  sidecar_trace_id text references sidecar_traces(id) on delete set null,
  memory_event_ids text[] not null default '{}',
  image_cue jsonb not null default '{}'::jsonb,
  image_job_id text references image_jobs(id) on delete set null,
  image_asset_ids text[] not null default '{}',
  metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null
);

create table if not exists image_assets (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  title text not null,
  source text not null,
  prompt text not null,
  negative_prompt text not null default '',
  safety_level text not null,
  character_ids text[] not null default '{}',
  tags text[] not null default '{}',
  job_id text references image_jobs(id) on delete set null,
  object_key text,
  thumbnail_object_key text,
  mime_type text,
  palette text[] not null default '{}',
  provider_metadata jsonb not null default '{}'::jsonb,
  representative boolean not null default false,
  feedback jsonb,
  created_at timestamptz not null
);

create table if not exists evaluation_scenarios (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  kind text not null,
  label text not null,
  query text not null,
  expected_signals text[] not null default '{}',
  source text not null,
  created_at timestamptz not null
);

create table if not exists audit_events (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  owner_id text not null references users(id),
  workspace_id text not null,
  project_id text not null,
  environment text not null,
  action text not null,
  resource_type text not null,
  resource_id text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null
);

create table if not exists redaction_requests (
  id text primary key,
  simulation_id text not null references simulations(id) on delete cascade,
  owner_id text not null references users(id),
  workspace_id text not null,
  project_id text not null,
  environment text not null,
  target_type text not null,
  target_id text not null,
  reason text not null,
  neural_map_node_ids text[] not null default '{}',
  status text not null,
  error text,
  created_at timestamptz not null,
  completed_at timestamptz
);

create table if not exists api_secret_refs (
  id text primary key,
  owner_id text not null references users(id) on delete cascade,
  provider text not null,
  display_label text not null,
  encrypted_secret bytea not null,
  encryption_key_id text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists prompt_modules_simulation_idx on prompt_modules(simulation_id);
create index if not exists prompt_modules_tags_idx on prompt_modules using gin(activation_tags);
create index if not exists chat_messages_session_idx on chat_messages(session_id, created_at);
create index if not exists memory_events_simulation_idx on memory_events(simulation_id, created_at);
create index if not exists session_handoffs_simulation_idx on session_handoffs(simulation_id, created_at);
create index if not exists continuity_checks_simulation_idx on continuity_checks(simulation_id, checked_at);
create index if not exists prompt_module_usages_turn_idx on prompt_module_usages(turn_id, created_at);
create index if not exists sidecar_traces_turn_idx on sidecar_traces(turn_id, created_at);
create index if not exists image_jobs_simulation_idx on image_jobs(simulation_id, created_at);
create index if not exists turn_traces_simulation_idx on turn_traces(simulation_id, created_at);
create index if not exists turn_traces_turn_idx on turn_traces(turn_id, created_at);
create index if not exists image_assets_simulation_idx on image_assets(simulation_id, created_at);
create index if not exists evaluation_scenarios_simulation_idx on evaluation_scenarios(simulation_id, kind);
create index if not exists audit_events_simulation_idx on audit_events(simulation_id, created_at);
create index if not exists audit_events_scope_idx on audit_events(owner_id, workspace_id, project_id, created_at);
create index if not exists redaction_requests_simulation_idx on redaction_requests(simulation_id, created_at);
