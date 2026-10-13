/* Build Foul Play's set-frequency prior from public mechanics, never test teams. */
'use strict';
const path = require('path');
const {Teams, toID} = require(path.resolve(process.argv[2], 'dist/sim'));
const generator = Teams.getGenerator('gen9randombattle', [179, 421, 991, 2027]);
const counts = {};
for (let i=0; i<100000; i++) {
  for (const p of generator.getTeam()) {
    const species = toID(p.species);
    const key = [p.level, toID(p.item), toID(p.ability), ...p.moves.map(toID).sort(), toID(p.teraType)].join(',');
    counts[species] ||= {};
    counts[species][key] = (counts[species][key] || 0) + 1;
  }
}
process.stdout.write(JSON.stringify(counts)+'\n');
