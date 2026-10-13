import Mathlib
import Lean
import Lean.Replay

open Lean Elab Command

/-- Replace only the sentinel term node, after separately parsing the entire proof. -/
private partial def replaceProof (s p : Syntax) : Syntax :=
  if s.isIdent && s.getId == `FRONTIER_PROOF_HOLE then p
  else s.setArgs (s.getArgs.map (fun a => replaceProof a p))

/-- Inspect parsed operations, not the spelling of identifiers such as `eval`. -/
private partial def checkProofSyntax (s : Syntax) : CommandElabM Unit := do
  match s with
  | .atom _ token =>
    if #["sorry", "admit", "axiom", "opaque", "unsafe", "partial", "native_decide",
         "bv_decide", "run_tac", "run_elab", "run_conv", "elab", "elab_rules",
         "by_elab", "eval", "eval%", "eval_expr", "run_meta", "macro", "macro_rules", "syntax",
         "initialize", "builtin_initialize", "set_option", "import", "open", "namespace",
         "section", "end", "attribute", "local", "scoped", "notation", "instance", "def",
         "theorem", "lemma", "constant", "extern", "implemented_by"].contains token then
      throwError "metaprogramming, command, option or axiom syntax: {token}"
  | .node _ _ args =>
    for arg in args do checkProofSyntax arg
  | _ => pure ()

elab "frontier_check " signature:str proof:str : command => do
  let initial ← getEnv
  let parse (category : Name) (s : String) : CommandElabM Syntax :=
    match Parser.runParserCategory initial category s with
    | .ok stx => pure stx
    | .error e => throwError "{e}"
  -- The signature comes from the organizer, and is elaborated independently.
  elabCommand (← parse `command s!"theorem FrontierExpected {signature.getString} := by sorry")
  let expected ← getConstInfo `FrontierExpected
  setEnv initial
  let term ← parse `term proof.getString
  checkProofSyntax term
  let command ← parse `command s!"theorem FrontierCandidate {signature.getString} := FRONTIER_PROOF_HOLE"
  elabCommand (replaceProof command term)
  if (← get).messages.hasErrors then throwError "proof elaboration failed"
  let submitted ← getEnv
  let mut added : Std.HashMap Name ConstantInfo := {}
  for (name, info) in submitted.constants.toList do
    if !(initial.contains name) then
      if info.isUnsafe || info.isPartial then throwError "unsafe/partial declaration"
      added := added.insert name info
  -- Recheck with the kernel in the saved environment. Never trust an elaborator's
  -- modified constants, options, or acceptance of a malformed proof expression.
  let replayed ← initial.replay added
  setEnv replayed
  let actual ← getConstInfo `FrontierCandidate
  unless actual.type == expected.type && actual.levelParams == expected.levelParams do
    throwError "the theorem type changed"
  let axioms ← collectAxioms `FrontierCandidate
  for axiomName in axioms do
    unless #[`propext, `Classical.choice, `Quot.sound].contains axiomName do
      throwError "disallowed axiom: {axiomName}"
  logInfo "FRONTIER_PROOF_VALID"
