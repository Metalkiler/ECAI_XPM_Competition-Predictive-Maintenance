
# Anticipating the Future: Predictive Maintenance for Public Transportation

## MOTIVATION
Predictive maintenance (PdM) plays a crucial role in today’s industrial landscape, supporting the principles of operational efficiency and environmental sustainability. In an era marked by increasing reliance on complex machinery and
equipment, the ability to foresee and prevent mechanical failures is of utmost importance. This importance is highlighted by the significant consequences that arise from equipment malfunctions during the regular operations of public
transportation vehicles. These consequences include a range of problems such as disrupted journeys for passengers and the resulting disappointment. These challenges not only affect the companies managing these transportation
services but also impact the communities they serve. The adoption of innovative predictive maintenance methods holds great promise, not only in reducing travel disruptions and service interruptions but also in significantly
enhancing the overall reliability and satisfaction associated with public transportation systems. Therefore, tackling this formidable challenge serves as a catalyst not only for technological innovation but also for the improvement of
transportation infrastructures, making them more dependable, efficient, and customer-focused.

## MATERIAL & METHODS
In the operational context of a metro train, data readings encompassing pressure, temperature, motor current, and air intake valve parameters were systematically gathered from the Air Production Unit (APU) of a compressor. The
dataset comprises a total of 15 essential features, which significantly contribute to establishing a comprehensive assessment of the health and functionality of the metro train system. This dataset serves as a valuable resource for
predicting system failures.
Within this context, our objective is to proactively predict the occurrence of both air and oil leaks. To ensure fairness in our approach, since datetime information about air leakages was provided but none was provided about oil
leakages , we employed and compared several unsupervised learning techniques, including Principal Component Analysis (PCA) coupled with K-means clustering, Isolation Forests, and Birch Clustering, to identify potential leaks.
This approach allowed us to evaluate and compare the effectiveness of unsupervised methods for both air and oil leakages, ensuring a fair assessment.
Then, we applied a Selective Sampling (SS) technique to eliminate sequences of leakage events, retaining only the initial occurrence from each sequence (Figure 1). This method results in multiple subsets of data, each consisting of a
sequence of normal working cycles that culminate in a leak event. The rationale behind this approach is to focus solely on when the leakage commenced, the contributing factors, and the feasibility of predicting its occurrence ahead
of time.

