/* Trusted/offline simulator adapter. Original code; upstream simulator stays unmodified. */
'use strict';
const path = require('path');
const readline = require('readline');
const root = path.resolve(process.argv[2]);
const {Battle, Dex, Teams} = require(path.join(root, 'dist/sim'));
const {extractChannelMessages} = require(path.join(root, 'dist/sim/battle'));
const FORMAT = 'gen9randombattle';
const FEATURE_NAMES = ['bias', 'damage', 'base_power', 'accuracy', 'priority', 'physical',
  'special', 'status', 'heal', 'setup', 'switch', 'tera', 'own_hp', 'foe_hp',
  'super_effective', 'resisted', 'switch_hp', 'switch_speed', 'switch_bulk', 'hazard'];
let games = [];

function visibleLog(battle, channel=0) {
  // Player channels include that player's exact HP, but mask the opponent's.
  // Never use omniscient channel -1. Both policies receive the same kind of view.
  return extractChannelMessages(battle.log.join('\n'), [channel])[channel]
    .filter(line => !/^\|(showteam|uhtml|uhtmlchange|html|debug|t:|seed)\|/.test(line));
}
function visibleActive(log, player) {
  const active = [null];
  for (const line of log) {
    const parts = line.split('|');
    const match = new RegExp('^' + player + '([ab]):').exec(parts[2] || '');
    if (!match) continue;
    const slot = match[1] === 'a' ? 0 : 1;
    if (['switch','drag','replace'].includes(parts[1])) {
      active[slot] = {species:parts[3].split(',')[0], condition:parts[4]};
    } else if (active[slot] && ['-damage','-heal','-sethp'].includes(parts[1])) {
      active[slot].condition = parts[3];
    } else if (active[slot] && parts[1] === 'faint') {
      active[slot].condition = '0 fnt';
    }
  }
  return active;
}
function hp(condition) {
  const m = /^(\d+)(?:\/(\d+))?/.exec(condition || '');
  return m ? Math.min(1, Number(m[1]) / Number(m[2] || 100)) : 0;
}
function legalActions(request) {
  if (request.wait) return [];
  if (request.teamPreview) throw Error('Random Battle must not have team preview');
  const mons = request.side.pokemon;
  const current = mons.find(p=>p.active);
  const replacements = mons.map((p,i)=>({p,i:i+1})).filter(({p})=>
    !p.active && p.condition.endsWith('fnt') === !!current?.reviving).map(({i})=>'switch '+i);
  if (request.forceSwitch) return replacements;
  if (request.active.length !== 1) throw Error('Expected a singles decision');
  const slot = request.active[0];
  const choices = slot.trapped ? [] : replacements;
  for (const [index, move] of slot.moves.entries()) {
    if (move.disabled || move.pp === 0) continue;
    choices.push(`move ${index+1}`);
    if (slot.canTerastallize) choices.push(`move ${index+1} terastallize`);
  }
  return choices;
}
function features(action, obs) {
  const f = Array(FEATURE_NAMES.length).fill(0); f[0] = 1;
  const mons = obs.request.side.pokemon, current = mons.find(p=>p.active);
  const foe = visibleActive(obs.public_log, obs.player === 'p1' ? 'p2' : 'p1')[0];
  f[12] = hp(current?.condition); f[13] = hp(foe?.condition);
  if (action.startsWith('switch')) {
    const target = mons[Number(action.split(' ')[1])-1];
    f[10] = 1; f[16] = hp(target.condition);
    f[17] = target.stats.spe/500;
    f[18] = (target.stats.def+target.stats.spd)/1000;
    return f;
  }
  const data = obs.request.active[0].moves[Number(action.split(' ')[1])-1];
  // Recharge is a forced simulator command, not a move in the public move dex.
  if (data.id === 'recharge') return f;
  const move = Dex.moves.get(data.id);
  if (!move.exists) throw Error('Unknown requested move: '+data.id);
  const own = Dex.species.get(current.details.split(',')[0]);
  const accuracy = move.accuracy === true ? 1 : move.accuracy/100;
  const power = move.basePower/100;
  const attacksFoe = !['self','allySide','allies','adjacentAlly','adjacentAllyOrSelf'].includes(move.target);
  f[2] = power; f[3] = accuracy; f[4] = move.priority/5;
  f[5] = +(move.category === 'Physical'); f[6] = +(move.category === 'Special');
  f[7] = +(move.category === 'Status'); f[11] = +action.includes('terastallize');
  f[8] = move.heal ? (1-f[12])*(move.heal[0]/move.heal[1]) : 0;
  f[9] = +(move.target === 'self' && !!move.boosts && Object.values(move.boosts).some(x=>x>0) && obs.turn <= 2);
  f[19] = +['stealthrock','spikes','toxicspikes','stickyweb'].includes(move.id) * +(obs.turn <= 2);
  if (foe && attacksFoe) {
    const species = Dex.species.get(foe.species);
    const effect = Dex.getImmunity(move.type,species) ? 2 ** Dex.getEffectiveness(move.type,species) : 0;
    f[14] = +(effect > 1); f[15] = +(effect < 1);
    f[1] = power * accuracy * effect * (own.types.includes(move.type) ? 1.5 : 1);
  }
  return f;
}
function observation(game, side) {
  const battle = game.battle, player = battle.sides[side];
  // Use the request already delivered to this player. Regenerating it can reveal
  // an unrevealed trapping ability before the player attempts a switch.
  if (battle.ended || !player.activeRequest || player.activeRequest.wait || player.isChoiceDone()) return null;
  const request = JSON.parse(JSON.stringify(player.activeRequest));
  const obs = {format:FORMAT, player:'p'+(side+1), turn:battle.turn, request, public_log:visibleLog(battle,side+1)};
  obs.legal_actions = legalActions(request);
  if (!obs.legal_actions.length) throw Error('Simulator produced an empty action menu');
  obs.action_features = obs.legal_actions.map(action=>features(action,obs));
  return obs;
}
function choose(obs, style, random) {
  if (style === 'random') return Math.floor(random()*obs.legal_actions.length);
  const weights = style === 'power' ? [0,0,2,0.1,0,0,0,0,0,0,-0.2,0.02] :
    style === 'type' ? [0,2,0,0.1,0.1,0,0,0,0,0,-0.4,0.03] :
    [0,2,0,0.1,1,0,0,-0.1,6,0.6,-0.4,0.03,0,0,0,0,0.2,0.01,0.01,0.7];
  let scores = obs.action_features.map(f => f.reduce((s,v,i)=>s+v*(weights[i]||0),0));
  if (style === 'stochastic') scores = scores.map(x=>x + random()*1.5);
  const best = Math.max(...scores), options = scores.flatMap((s,i)=>Math.abs(s-best)<1e-9 ? [i] : []);
  return options[Math.floor(random()*options.length)];
}
function createBattle(c) {
  if (Object.hasOwn(c, 'teams')) {
    // Fixed evaluation sets contain the full upstream-generated teams. Never
    // fall back to random generation if stored data is absent or malformed.
    if (!Array.isArray(c.teams) || c.teams.length !== 2 || c.teams.some(team => {
      if (typeof team !== 'string') return true;
      const sets = Teams.unpack(team);
      return !sets || sets.length !== 6 || sets.some(p => !Dex.species.get(p.species).exists ||
        !p.moves.length || p.moves.length > 4 || p.moves.some(m => !Dex.moves.get(m).exists));
    })) throw Error('Invalid stored evaluation teams');
    return new Battle({formatid:FORMAT, seed:c.seed, p1:{name:'Player 1',team:c.teams[0]},
      p2:{name:'Player 2',team:c.teams[1]}, strictChoices:false});
  }
  // Let the unmodified upstream format generate both teams and balancing levels.
  // Explicit team seeds make paired games reproducible without publishing teams.
  return new Battle({formatid:FORMAT, seed:c.seed, p1:{name:'Player 1',seed:c.team_seeds[0]},
    p2:{name:'Player 2',seed:c.team_seeds[1]}, strictChoices:false});
}
function applyChoice(battle, side, action) {
  const before = JSON.stringify(battle.sides[side].activeRequest);
  const accepted = battle.choose('p'+(side+1), action);
  if (!accepted) {
    const request = battle.sides[side].activeRequest;
    // Standard Showdown semantics: hidden trapping/disabled moves can reveal new
    // information and ask this player again. Other accepted choices stay locked.
    if (!request?.update || JSON.stringify(request) === before) {
      throw Error('Simulator rejected an offered action: '+battle.sides[side].choice.error);
    }
  }
  return accepted;
}
function rng(seed) { let n = seed >>> 0; return () => {n=(Math.imul(1664525,n)+1013904223)>>>0;return n/4294967296;}; }
function newGame(c) {
  return {battle:createBattle(c), candidate:c.seat, opponent:c.opponent,
    random:rng(c.policy_seed), candidateRandom:rng(c.policy_seed ^ 0xa564cb81), rounds:0, pending:null, result:null};
}
function init(configs) {
  if (!Array.isArray(configs) || configs.length > 32) throw Error('At most 32 active games');
  for (const game of games) game.battle.destroy();
  games = configs.map(newGame);
}
function refill(replacements) {
  const slots = new Set();
  // Validate the entire command before replacing any game. Other slots retain
  // their pending requests, accepted choices, battle RNG and public history.
  for (const {slot} of replacements) {
    if (!Number.isInteger(slot) || !games[slot]?.result || slots.has(slot)) {
      throw Error('Can only refill distinct completed game slots');
    }
    slots.add(slot);
  }
  for (const {slot, config} of replacements) {
    games[slot].battle.destroy();
    games[slot] = newGame(config);
  }
  return state();
}
function state() {
  const observations=[], indices=[], opponents=[];
  games.forEach((g,i) => {
    if (g.result) return;
    if (g.battle.ended) {
      g.result={points:g.battle.winner ? +(g.battle.winner === 'Player '+(g.candidate+1)) : 0.5,
        reason:'finished', turns:g.battle.turn, rounds:g.rounds}; return;
    }
    if (g.battle.turn >= 200 || g.rounds >= 512) {
      g.result={points:0,reason:'cap',turns:g.battle.turn,rounds:g.rounds}; return;
    }
    if (!g.pending) {
      const own=observation(g,g.candidate), other=observation(g,1-g.candidate);
      g.pending={own, other};
    }
    if (g.pending.own) { observations.push(g.pending.own); indices.push(i); }
    if (g.pending.other) opponents.push({game:i,decision:g.rounds,observation:g.pending.other});
  });
  // Only the observations field is sent to the candidate endpoint.
  return {observations,indices,opponents,results:games.map(g=>g.result)};
}
function step(actions, opponentActions) {
  const current=state();
  if (actions.length !== current.indices.length) throw Error('Wrong number of actions');
  if (!Array.isArray(opponentActions) || opponentActions.length !== current.opponents.length) throw Error('Missing Foul Play choices');
  const choices=new Map();
  const opponentChoices=new Map();
  const forfeits=new Set();
  current.opponents.forEach((item,i)=>{
    const action=opponentActions[i], menu=item.observation.legal_actions;
    // -1: Foul Play made an illegal choice and forfeits this game; the candidate wins it.
    if (action === -1) { forfeits.add(item.game); return; }
    if (!Number.isInteger(action) || action<0 || action>=menu.length) throw Error('Invalid Foul Play action');
    opponentChoices.set(item.game,menu[action]);
  });
  current.indices.forEach((index,i) => {
    const obs=games[index].pending.own, action=actions[i];
    if (!Number.isInteger(action) || action<0 || action>=obs.legal_actions.length) throw Error('Invalid action index');
    choices.set(index,obs.legal_actions[action]);
  });
  games.forEach((g,i) => {
    if (g.result) return;
    if (forfeits.has(i)) {
      g.result={points:1,reason:'opponent_illegal',turns:g.battle.turn,rounds:g.rounds}; g.pending=null; return;
    }
    const pending=g.pending;
    // Both choices are fixed before either is applied. Opponent never sees our pending order.
    if (pending.own) applyChoice(g.battle,g.candidate,choices.get(i));
    if (pending.other) applyChoice(g.battle,1-g.candidate,opponentChoices.get(i));
    g.pending=null; g.rounds++;
  });
  return state();
}
function handle(message) {
  if (message.op === 'init') {init(message.games);return state();}
  if (message.op === 'refill') return refill(message.games);
  if (message.op === 'step') return step(message.actions, message.opponent_actions);
  if (message.op === 'baseline') {
    const s=state(); return {actions:s.observations.map((o,i)=>choose(o,message.style,games[s.indices[i]].candidateRandom))};
  }
  if (message.op === 'features') return {names:FEATURE_NAMES};
  throw Error('Unknown simulator operation');
}
if (require.main === module) {
  const input=readline.createInterface({input:process.stdin});
  input.on('line', line => {
    try {process.stdout.write(JSON.stringify(handle(JSON.parse(line)))+'\n');}
    catch (e) {process.stdout.write(JSON.stringify({error:e.stack})+'\n');process.exitCode=1;input.close();}
  });
}
module.exports={legalActions, visibleLog, visibleActive, observation, features, choose, createBattle, applyChoice, init, refill, state, step};
