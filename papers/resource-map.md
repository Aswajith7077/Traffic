

# 1. Traffic management for emv priority on Visual Sensing

`references/Traffic-management-for-emv-priority-on-visual-sensing.md`

**Problem addressed:** Why emergency vehicle consideration is important in traffic congestion control. Hilight uses the MARL traffic congestion control, without considering the real world uncertainity. Why this extention on EMV is required and consideration would serve some part of the uncertainity in the real world scenario, more certain in some metrics. 

**Solution Addressed by the Research Paper:**

1. Motivation for EMV consideration — strong support

The paper establishes that fixed green light sequences, which most MARL systems also implicitly assume as the action space, are determined without EMV presence. This is precisely the uncertainty gap you're pointing to.

2. Quantified real-world cost of ignoring EMVs

~700 annual fatalities in Ireland from late ambulance responses
4,500 annual US ambulance crashes; 662 deaths over 20 years
31,600 fire vehicle accidents over 10 years

These statistics directly justify why ignoring EMVs constitutes a meaningful real-world uncertainty, not a fringe case.

3. Metric-level evidence where EMV consideration improves performance

The paper shows measurable gains when EMV priority is incorporated:

End-to-end delay reduced by up to 60% at 100 nodes vs IEEE 802.11p
Throughput improvement of ~70% at 100 nodes


# 2. 
