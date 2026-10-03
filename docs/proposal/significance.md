# Significance (draft for the README)

> **Draft** by Michelle (Business stream), from the team Google Doc, received 2026-10-03. The text
> below is hers, with one typo fixed ("undroogued"). Review notes and suggested fixes follow it.
> Sources: `docs/literature/business_literature_review.md`.

## Why is the problem important/significant?

The Benguela-South Atlantic region supports activities such as fisheries, shipping, oil and gas,
diamond mining, and aquaculture. These activities can be affected by the movement of material at
the ocean surface. For example, if oil or other floating material is released into the ocean, its
subsequent location depends partly on the surface transport. Therefore, identifying the areas that
are connected to the Benguela region can help determine where floating material originating within
the region may subsequently travel.

This makes the question of "where does a drifter end up?" practically useful. By estimating
transition probabilities between spatial cells, our project can identify the dominant directions
of surface transport and the regions most frequently reached by drifters originating within our
study area. Furthermore, by comparing drogued and undrogued drifters, we can assess whether the
observed transport is more representative of general subsurface circulation or the movement of
surface-floating material such as oil.

## AI Agent Review notes (take with a grain of salt as unsure if it could cross-reference accurately due to permissions) (2026-10-03)

1. **Cite the activities list:** Benguela Current Convention (n.d.). The BCC page also says major
   international shipping routes cross the region and that this is a pollution risk. That's a
   stronger, sourced reason than the general list, so consider leading with it.
2. **Scope of "regions most frequently reached".** Our P only follows drifters inside the box;
   leaving is a single "left R" state (decision 11b). Suggested: "...the parts of the
   Benguela–South Atlantic box most often reached from each starting cell, and how much material
   leaves the box altogether."
3. **The drogue sentence needs correcting.** Drogued drifters follow the current a few metres
   down (the drogue hangs below the float, so it is not "general subsurface circulation").
   Undrogued drifters also feel the wind and the fast top layer. Our planned check is also
   different: drogued-only data is too thin to estimate P (9% of 4 deg cells adequate), so we
   compare all drifters against undrogued drifters only at 4 deg (decision 6a). Suggested:
   "Because many drifters have lost their drogue and so also feel the wind, our main estimate
   describes near-surface transport including wind effects, which suits floating material such as
   oil. We test how much this matters by comparing it with an estimate from undrogued drifters
   only." Add the drogue depth from the GDP documentation (`gdp-overview.pdf`); it isn't confirmed
   here.
4. **Missing for the marking criteria (significance to the client, argued with literature):**
   - the offshore oil exploration point from plan section 2 (recent Namibia/Angola discoveries),
     with a source;
   - why the result matters to Shane Elipot / the GDP (e.g. a reusable, tested matrix with
     uncertainty for one region, built from his dataset);
   - at least one citation in this section itself.
