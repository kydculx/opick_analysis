#!/bin/sh
# 맥용 원키 학습 스크립트 (11개 리그)
#   sh ml/train.sh                              → 메뉴 (리그→방법)
#   sh ml/train.sh permatch <리그> [--new]      → autotune (생략시 프리미어)
#   <리그>는 약어 가능: epl k1 k2 j1 bl ll l1 sa ered mls ale (대소문자 무관)
#   sh ml/train.sh fast --league <리그> --train <시즌> --valid <시즌> --ver <버전> [--new]
#   sh ml/train.sh auto <리그> --tune <시즌,콤마> [--new] → 랜덤탐색(웹과 동일)
#   sh ml/train.sh grid <리그> --tune <시즌,콤마> --grid-step 0.5 [--new] → 전수탐색
# 예: sh ml/train.sh permatch bundesliga --new
set -e
cd "$(dirname "$0")/.."
set -a; . ./.env.local 2>/dev/null || true; set +a
PY="ml/.venv/bin/python"
# 병렬 워커 수: 기본 CPU 코어 수 (넘파이 스레드 경합 방지용 스레드 제한과 함께 사용)
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# numba 스레드: autotune 워커는 단일 스레드(과다점유 방지), grid 본체는 코어 수
NUMBA_NUM_THREADS=1
export NUMBA_NUM_THREADS
# 연속 미개선 trial 수 제한 (0=끄기). 새학습 장기 탐색 시 정체 구간 절약용.
PATIENCE="${PATIENCE:-0}"

leagues="premier_league k_league_1 k_league_2 j1_league bundesliga laliga ligue_1 serie_a eredivisie mls a_league"

preset_for() {
  case "$1" in
    premier_league|bundesliga|eredivisie|laliga|ligue_1|serie_a|a_league)
      TRAIN="2016-2017,2017-2018,2018-2019,2019-2020,2020-2021,2021-2022"
      VALID="2022-2023" ;;
    k_league_1|k_league_2|mls)
      TRAIN="2016,2017,2018,2019,2020,2021"
      VALID="2022" ;;
    j1_league)
      TRAIN="2017,2018,2019,2020,2021,2022"
      VALID="2023" ;;
    *) echo "알 수 없는 리그: $1 (가능: $leagues)"; exit 2 ;;
  esac
}

ALL_FEATS="rank,power,hstr,cond,att,def,val,form5,h2h5,avg_goals,avg_conceded,avg_poss,market"
FIVE_FEATS="rank,power,val,form5,market"

ver_hash() {
  if command -v shasum >/dev/null 2>&1; then printf '%s' "$1" | shasum -a 1 | cut -c1-4
  elif command -v sha1sum >/dev/null 2>&1; then printf '%s' "$1" | sha1sum | cut -c1-4
  else printf '%s' "$1" | md5 | awk '{print $NF}' | cut -c1-4; fi
}

make_ver() {
  _n=$(printf '%s' "$1" | awk -F',' '{print NF}')
  _m=$(printf '%s' "$2" | awk -F',' '{print NF}')
  _h=$(ver_hash "$1|$2")
  printf 'tr%s-f%s-x%s' "$_n" "$_m" "$_h"
}

league_ko() {
  case "$1" in
    premier_league) echo "프리미어리그" ;;
    k_league_1) echo "K리그 1" ;;
    k_league_2) echo "K리그 2" ;;
    j1_league) echo "J1리그" ;;
    bundesliga) echo "분데스리가" ;;
    laliga) echo "라리가" ;;
    ligue_1) echo "리그1(프랑스)" ;;
    serie_a) echo "세리에A" ;;
    eredivisie) echo "에레디비시" ;;
    mls) echo "MLS" ;;
    a_league) echo "A리그(호주)" ;;
    *) echo "$1" ;;
  esac
}

league_alias() {
  a=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  case "$a" in
    epl) echo "premier_league" ;;
    k1) echo "k_league_1" ;;
    k2) echo "k_league_2" ;;
    j1|jleague) echo "j1_league" ;;
    bl|bundes|bundesliga) echo "bundesliga" ;;
    ll|laliga) echo "laliga" ;;
    l1|ligue1|ligue_1) echo "ligue_1" ;;
    sa|seriea|serie_a) echo "serie_a" ;;
    ered|eredivisie) echo "eredivisie" ;;
    mls) echo "mls" ;;
    ale|aleague|a_league) echo "a_league" ;;
    *) echo "$1" ;;
  esac
}

league_short() {
  case "$1" in
    premier_league) echo "EPL" ;;
    k_league_1) echo "K1" ;;
    k_league_2) echo "K2" ;;
    j1_league) echo "J1" ;;
    bundesliga) echo "BL" ;;
    laliga) echo "LL" ;;
    ligue_1) echo "L1" ;;
    serie_a) echo "SA" ;;
    eredivisie) echo "ERE" ;;
    mls) echo "MLS" ;;
    a_league) echo "ALE" ;;
    *) echo "$1" ;;
  esac
}

pick_league() {
  i=1
  for l in $leagues; do echo "  $i) [$(league_short "$l")] $(league_ko "$l") ($l)" >&2; i=$((i + 1)); done
  printf "리그 번호 [1]: " >&2
  read -r n
  n="${n:-1}"
  i=1
  for l in $leagues; do
    if [ "$i" = "$n" ]; then echo "$l"; return; fi
    i=$((i + 1))
  done
  echo "premier_league"
}

run_permatch() {
  league="$1"; extra="$2"; trials="${3:-10000}"
  preset_for "$league"
  case "$extra" in
    *--features*) FEATS=$(printf '%s' "$extra" | sed -n 's/.*--features[ =]\([^ ]*\).*/\1/p') ;;
    *--five*) extra=$(printf '%s' "$extra" | sed 's/--five//g'); FEATS="$FIVE_FEATS"; extra="$extra --features $FIVE_FEATS" ;;
    *) FEATS="$ALL_FEATS" ;;
  esac
  [ -z "$FEATS" ] && FEATS="$ALL_FEATS"
  VER=$(make_ver "$TRAIN" "$FEATS")
  case "$extra" in
    *--features*) echo "→ permatch autotune: [$(league_short "$league")] train=[$TRAIN] valid=[미학습전체] ver=$VER feats=[$FEATS] trials=$trials" ;;
    *) echo "→ permatch autotune: [$(league_short "$league")] train=[$TRAIN] valid=[미학습전체] ver=$VER trials=$trials $extra" ;;
  esac
  # shellcheck disable=SC2086
  exec $PY ml/permatch_mode.py --league "$league" --train "$TRAIN" \
    --valid "auto" --ver "$VER" --trials "$trials" --jobs "$JOBS" --patience "$PATIENCE" $extra
}

menu() {
  echo ""
  echo "학습할 리그를 선택하세요."
  league=$(pick_league)
  preset_for "$league"
  echo ""
  echo "학습방법을 선택하세요."
  echo "  1) 경기당 새학습"
  echo "  2) 경기당 계속학습"
  echo "  3) 경기당 랜덤탐색(웹과 동일)"
  echo "  4) 모든 경우수 조회(전수탐색)"
  echo "  5) 5피처 새학습 (rank,power,val,form5,market)"
  echo "  6) 5피처 계속학습 (rank,power,val,form5,market)"
  echo "  7) 5피처 랜덤탐색(웹과 동일)"
  echo "  8) 5피처 전수탐색"
  echo "  0) 종료"
  printf "번호: "
  read -r m
  case "$m" in
    1) printf "학습 횟수 [기본 10000]: "; read -r trials
       case "$trials" in ""|*[!0-9]*) trials=10000 ;; esac
       run_permatch "$league" "--new" "$trials" ;;
    2) printf "학습 횟수 [기본 10000]: "; read -r trials
       case "$trials" in ""|*[!0-9]*) trials=10000 ;; esac
       run_permatch "$league" "" "$trials" ;;
    3) printf "조절 시즌(콤마, 비우면 미학습 전체): "; read -r ts
       VER=$(make_ver "$TRAIN" "$ALL_FEATS")
        echo "→ 랜덤탐색: [$(league_short "$league")] ver=$VER tune=[${ts:-미학습 전체}] (Ctrl+C로 중지)"
       # shellcheck disable=SC2086
       exec $PY ml/permatch_mode.py --mode auto --league "$league" --ver "$VER" --tune "$ts" ;;
    4) printf "조절 시즌(콤마, 비우면 미학습 전체): "; read -r ts
       printf "축 간격 [기본 0.5]: "; read -r gs
       gs="${gs:-0.5}"
       VER=$(make_ver "$TRAIN" "$ALL_FEATS")
       echo "→ 전수탐색: [$(league_short "$league")] ver=$VER tune=[${ts:-미학습 전체}] step=$gs (Ctrl+C로 중지, 계속학습 지원)"
       # shellcheck disable=SC2086
       NUMBA_NUM_THREADS="$JOBS" exec $PY ml/permatch_mode.py --mode grid --league "$league" --ver "$VER" --tune "$ts" --grid-step "$gs" --jobs "$JOBS" ;;
    5) printf "학습 횟수 [기본 10000]: "; read -r trials
       case "$trials" in ""|*[!0-9]*) trials=10000 ;; esac
       run_permatch "$league" "--new --five" "$trials" ;;
    6) printf "학습 횟수 [기본 10000]: "; read -r trials
       case "$trials" in ""|*[!0-9]*) trials=10000 ;; esac
       run_permatch "$league" "--five" "$trials" ;;
    7) printf "조절 시즌(콤마, 비우면 미학습 전체): "; read -r ts
       VER=$(make_ver "$TRAIN" "$FIVE_FEATS")
        echo "→ 5피처 랜덤탐색: [$(league_short "$league")] ver=$VER tune=[${ts:-미학습 전체}] (Ctrl+C로 중지)"
       # shellcheck disable=SC2086
       exec $PY ml/permatch_mode.py --mode auto --league "$league" --ver "$VER" --features "$FIVE_FEATS" --tune "$ts" ;;
    8) printf "조절 시즌(콤마, 비우면 미학습 전체): "; read -r ts
       printf "축 간격 [기본 0.5]: "; read -r gs
       gs="${gs:-0.5}"
       VER=$(make_ver "$TRAIN" "$FIVE_FEATS")
       echo "→ 5피처 전수탐색: [$(league_short "$league")] ver=$VER tune=[${ts:-미학습 전체}] step=$gs (Ctrl+C로 중지, 계속학습 지원)"
       # shellcheck disable=SC2086
       NUMBA_NUM_THREADS="$JOBS" exec $PY ml/permatch_mode.py --mode grid --league "$league" --ver "$VER" --features "$FIVE_FEATS" --tune "$ts" --grid-step "$gs" --jobs "$JOBS" ;;
    *) exit 0 ;;
  esac
}

if [ $# -eq 0 ]; then menu; exit 0; fi
cmd="$1"; shift
case "$cmd" in
  permatch) league="premier_league"
    case "${1:-}" in ""|-*) : ;; *) league="$1"; shift ;; esac
    league=$(league_alias "$league")
    run_permatch "$league" "$*" ;;
  permatch-new) league="premier_league"
    case "${1:-}" in ""|-*) : ;; *) league="$1"; shift ;; esac
    league=$(league_alias "$league")
    run_permatch "$league" "--new $*" ;;
  permatch5|p5) league="premier_league"
    case "${1:-}" in ""|-*) : ;; *) league="$1"; shift ;; esac
    league=$(league_alias "$league")
    run_permatch "$league" "--five $*" ;;
  k1) run_permatch "k_league_1" "$*" ;;
  k1-new) run_permatch "k_league_1" "--new $*" ;;
  fast) exec $PY ml/permatch_mode.py --fast "$@" ;;
  auto) league="k_league_1"
    case "${1:-}" in ""|-*) : ;; *) league="$1"; shift ;; esac
    league=$(league_alias "$league")
    preset_for "$league"
    VER=$(make_ver "$TRAIN" "$ALL_FEATS")
    echo "→ 랜덤탐색: [$(league_short "$league")] ver=$VER (tune 생략시 미학습 전체, Ctrl+C로 중지)"
    # shellcheck disable=SC2086
    exec $PY ml/permatch_mode.py --mode auto --league "$league" --ver "$VER" $* ;;
  grid) league="k_league_1"
    case "${1:-}" in ""|-*) : ;; *) league="$1"; shift ;; esac
    league=$(league_alias "$league")
    preset_for "$league"
    VER=$(make_ver "$TRAIN" "$ALL_FEATS")
    echo "→ 전수탐색: [$(league_short "$league")] ver=$VER (tune 생략시 미학습 전체, Ctrl+C로 중지)"
    # shellcheck disable=SC2086
    NUMBA_NUM_THREADS="$JOBS" exec $PY ml/permatch_mode.py --mode grid --league "$league" --ver "$VER" --jobs "$JOBS" $* ;;
  auto5) league="k_league_1"
    case "${1:-}" in ""|-*) : ;; *) league="$1"; shift ;; esac
    league=$(league_alias "$league")
    preset_for "$league"
    VER=$(make_ver "$TRAIN" "$FIVE_FEATS")
    echo "→ 5피처 랜덤탐색: [$(league_short "$league")] ver=$VER (tune 생략시 미학습 전체, Ctrl+C로 중지)"
    # shellcheck disable=SC2086
    exec $PY ml/permatch_mode.py --mode auto --league "$league" --ver "$VER" --features "$FIVE_FEATS" $* ;;
  grid5) league="k_league_1"
    case "${1:-}" in ""|-*) : ;; *) league="$1"; shift ;; esac
    league=$(league_alias "$league")
    preset_for "$league"
    VER=$(make_ver "$TRAIN" "$FIVE_FEATS")
    echo "→ 5피처 전수탐색: [$(league_short "$league")] ver=$VER (tune 생략시 미학습 전체, Ctrl+C로 중지)"
    # shellcheck disable=SC2086
    NUMBA_NUM_THREADS="$JOBS" exec $PY ml/permatch_mode.py --mode grid --league "$league" --ver "$VER" --features "$FIVE_FEATS" --jobs "$JOBS" $* ;;
  *) menu ;;
esac
