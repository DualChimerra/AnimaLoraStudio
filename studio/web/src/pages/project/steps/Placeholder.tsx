/** PP2-PP6 的步骤页占位。每个 PP 落地时会被实际页面替换。 */
export default function StepPlaceholder({
  step,
  doc,
}: {
  step: string
  doc: string
}) {
  return (
    <div className="max-w-xl mt-8 space-y-3">
      <h2 className="text-lg font-semibold">{step}</h2>
      <p className="text-sm text-fg-secondary">
        This step will be implemented in the <code className="text-accent">{doc}</code> stage.
      </p>
      <p className="text-xs text-fg-tertiary">
        Only the Project / Version data model and navigation are implemented so far (PP1);
        next up is PP2 download integration.
      </p>
    </div>
  )
}
