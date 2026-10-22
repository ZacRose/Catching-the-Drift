# Background

> **Draft** by Michelle (Business stream), from the team Google Doc, received 2026-10-03. The text
> below is hers, unchanged. Review notes and suggested fixes follow it. Sources and full references:
> `docs/literature/business_literature_review.md`.
>
> **Status (2026-10-21):** review notes 1-4 were applied in the submitted proposal (sections 1-2),
> which is the corrected, cited version to build the report background from.

## Intro/background and existing studies/solutions

The Benguela Current is one of the world's major eastern boundary upwelling systems and consists
of several interconnected regions, including the Angolan, northern Benguela, southern Benguela,
and Agulhas Bank regions. The system is characterised by strong upwelling, offshore transport,
ocean circulation, and substantial spatial and temporal variability. Hutchings et al. (2009)
provide a multidisciplinary overview of the Benguela system and highlight the role of regional
circulation, offshore transport, and larger-scale oceanographic processes in shaping its physical
and ecological characteristics.

A previous study has used drifter observations to investigate surface transport and connectivity
in the South Atlantic. Drouin and Lozier (2019) used Global Drifter Program trajectories together
with OSCAR surface-current data and subsurface float observations to investigate surface pathways
and the origins of water entering the Benguela Current. Their results demonstrate complex
transport pathways connecting different regions of the South Atlantic. In addition, Novelli et al.
(2020) compared drogued and undrogued drifters with crude oil transport and found that undrogued
drifters showed more similar spreading behaviour to oil slicks.

## AI Agent Review notes (take with a grain of salt as it cross-reference limited by permissions) (2026-10-03)

1. **Drouin and Lozier: remove "and subsurface float observations".** Their abstract names only
   GDP drifters and OSCAR. Suggested replacement for that paragraph's middle sentences:
   "Drouin and Lozier (2019) used the same Global Drifter Program data as this project, with the
   OSCAR surface-current product, to trace where water entering the Benguela Current comes from.
   Their backward trajectories suggest a sizeable share arrives from the Pacific via Drake
   Passage, while only a small part of the Agulhas water leaks into the South Atlantic."
   (Check against the full paper first.)
2. **Novelli et al.: add the caveat.** Suggested: "...found that patches of undrogued drifters
   spread like crude oil slicks, and reached the coast far more often than drogued ones (about
   25% vs 5%) in the Gulf of Mexico. Those were probably a different drifter design from the GDP
   buoys used here, so we treat this as support for, not proof of, using undrogued drifters as an
   oil proxy." This matches decision 6a in `project_plan.txt`.
3. **"A previous study" understates the literature.** Drifter-based transition matrices are an
   established method (van Sebille et al. 2012; Maximenko et al. 2012; McAdam & van Sebille
   2018). One or two sentences on them would show the method is feasible and what our project
   adds: a Benguela-focused matrix with uncertainty, and a check on how it changes with cell size.
   These were found by an AI search from abstracts and need reading before citing.
4. **Missing link to our region box.** The box (lat −42 to −4, lon −60 to 20) reaches well beyond
   the Benguela, to the Brazil–Malvinas Confluence and the Agulhas retroflection. One sentence on
   why (section 2 of the plan: basin-scale connectivity, not just the coast) would tie this
   background to our scope.
