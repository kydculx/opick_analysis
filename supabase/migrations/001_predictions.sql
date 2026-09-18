create table if not exists public.predictions (
  id bigint generated always as identity primary key,
  league_code text not null,
  season text not null,
  source_match_id text not null,
  model_ver text not null default 'v1',
  prob_home double precision not null,
  prob_draw double precision not null,
  prob_away double precision not null,
  created_at timestamptz not null default now(),
  unique (league_code, season, source_match_id, model_ver)
);

create index if not exists predictions_league_season_idx
  on public.predictions (league_code, season);
