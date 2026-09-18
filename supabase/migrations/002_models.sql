create table if not exists public.models (
  ver text primary key,
  league_code text not null,
  train_seasons text[] not null default '{}',
  model_type text not null default 'logreg',
  metrics jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists models_league_idx on public.models (league_code);
