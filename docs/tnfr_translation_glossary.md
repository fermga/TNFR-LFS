# TNFR × LFS Spanish → English Glossary

> Developer note: Use this glossary when migrating recommender and setup plan messaging to English. Terminology aligns with existing TNFR vocabulary (entry/apex/exit, Sense Index, ΔNFR, etc.).

## Recommender rules (`tnfr_lfs/recommender/rules.py`)

| Spanish phrase | Canonical English replacement |
| --- | --- |
| Basic Setup Guide · Frenada óptima [BAS-FRE] | Basic Setup Guide · Optimal Braking [BAS-FRE] |
| Advanced Setup Guide · Barras estabilizadoras [ADV-ARB] | Advanced Setup Guide · Anti-Roll Bars [ADV-ARB] |
| Advanced Setup Guide · Configuración de diferenciales [ADV-DIF] | Advanced Setup Guide · Differential Configuration [ADV-DIF] |
| Basic Setup Guide · Uso de pianos [BAS-CUR] | Basic Setup Guide · Kerb Usage [BAS-CUR] |
| Advanced Setup Guide · Alturas y reparto de carga [ADV-RDH] | Advanced Setup Guide · Ride Heights & Load Distribution [ADV-RDH] |
| Basic Setup Guide · Balance aerodinámico [BAS-AER] | Basic Setup Guide · Aero Balance [BAS-AER] |
| Basic Setup Guide · Constancia de pilotaje [BAS-DRV] | Basic Setup Guide · Consistent Driving [BAS-DRV] |
| Advanced Setup Guide · Presiones y caídas [ADV-TYR] | Advanced Setup Guide · Pressures & Camber [ADV-TYR] |
| Advanced Setup Guide · Amortiguadores [ADV-DMP] | Advanced Setup Guide · Dampers [ADV-DMP] |
| Advanced Setup Guide · Rigidez de muelles [ADV-SPR] | Advanced Setup Guide · Spring Stiffness [ADV-SPR] |
| Prioriza el bias de frenos (proyección ∇NFR∥) | Prioritize brake bias (∇NFR∥ projection) |
| Refuerza el bloqueo de retención (proyección ∇NFR∥) | Reinforce coast locking (∇NFR∥ projection) |
| Ajusta las barras estabilizadoras (proyección ∇NFR⊥) | Tune the anti-roll bars (∇NFR⊥ projection) |
| Afina el toe delantero (proyección ∇NFR⊥) | Fine-tune front toe (∇NFR⊥ projection) |
| Afina el toe trasero (proyección ∇NFR⊥) | Fine-tune rear toe (∇NFR⊥ projection) |
| neumáticos | Tyres |
| suspensión | Suspension |
| chasis | Chassis |
| frenos | Brakes |
| transmisión | Transmission |
| pista | Track |
| piloto | Driver |
| abrir toe delantero | Increase front toe-out |
| cerrar toe delantero | Reduce front toe-out |
| desplazar el balance de frenos hacia delante | Shift brake bias forward |
| desplazar el balance de frenos hacia atrás | Shift brake bias rearward |
| endurecer el rebote delantero | Stiffen front rebound |
| ablandar el rebote delantero | Soften front rebound |
| endurecer la barra estabilizadora | Stiffen the anti-roll bar |
| ablandar la barra estabilizadora | Soften the anti-roll bar |
| aumentar precarga delantera | Increase front preload |
| reducir precarga delantera | Reduce front preload |
| subir presión en neumáticos exteriores | Raise pressure in outside tyres |
| bajar presión en neumáticos exteriores | Lower pressure in outside tyres |
| cerrar el diferencial de aceleración | Increase power locking |
| abrir el diferencial de aceleración | Reduce power locking |
| incrementar caída trasera | Increase rear camber |
| reducir caída trasera | Reduce rear camber |
| endurecer compresión trasera | Stiffen rear compression |
| ablandar compresión trasera | Soften rear compression |
| {delta:+.1f}% bias delante | {delta:+.1f}% brake bias forward |
| {delta:+.1f} N/mm muelle delantero (νf_susp · ∇NFR⊥) | {delta:+.1f} N/mm front spring (νf_susp · ∇NFR⊥) |
| {delta:+.0f} clicks rebote delante | {delta:+.0f} front rebound clicks |
| {delta:+.0f} clicks compresión delante | {delta:+.0f} front compression clicks |
| {delta:+.1f} psi eje delantero | {delta:+.1f} psi front axle |
| {delta:+.1f}° camber delantero | {delta:+.1f}° front camber |
| {delta:+.1f}° caster | {delta:+.1f}° caster |
| {delta:+.2f}° toe delantero | {delta:+.2f}° front toe |
| {delta:+.0f} pasos barra delantera | {delta:+.0f} front anti-roll bar steps |
| {delta:+.1f} psi exteriores | {delta:+.1f} psi outside tyres |
| {delta:+.0f} clicks rebote trasero | {delta:+.0f} rear rebound clicks |
| {delta:+.0f} pasos barra trasera | {delta:+.0f} rear anti-roll bar steps |
| {delta:+.1f}° camber trasero | {delta:+.1f}° rear camber |
| {delta:+.0f}% LSD potencia | {delta:+.0f}% LSD power |
| {delta:+.1f} N/mm muelle trasero (νf_susp · ∇NFR⊥) | {delta:+.1f} N/mm rear spring (νf_susp · ∇NFR⊥) |
| {delta:+.1f} mm altura trasera | {delta:+.1f} mm rear ride height |
| {delta:+.1f} psi eje trasero | {delta:+.1f} psi rear axle |
| {delta:+.0f}% LSD retención | {delta:+.0f}% LSD coast |
| {delta:+.2f}° toe trasero | {delta:+.2f}° rear toe |
| {delta:+.1f}° ala trasera | {delta:+.1f}° rear wing |

## Setup plan exporter (`tnfr_lfs/exporters/setup_plan.py`)

| Spanish phrase | Canonical English replacement |
| --- | --- |
| Entrada | Entry |
| Vértice | Apex |
| Salida | Exit |
| Fase | Phase |
