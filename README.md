# Active Learning for Improving Out-of-Distribution Lab-in-the-Loop Experimental Design
This project explores how active learning can be used to improve antibody-antigen binding prediction in library-on-library screening settings, where many antibodies are tested against many antigens. Predicting binding in such scenarios is especially challenging when both antibodies and antigens in the test set are out-of-distribution (i.e., not seen during training).

We developed and evaluated 14 novel active learning strategies using the Absolut! simulation framework to iteratively select the most informative antibody-antigen pairs for labeling. Compared to random selection, our best-performing strategy:

* Reduced the number of required antigen mutants by up to 35%

* Sped up the learning process by 28 steps

These results suggest that active learning can significantly enhance experimental design and reduce the cost of generating binding data in many-to-many screening setups.

📄 Preprint available: [bioRxiv 2025.02.26.640110](https://www.biorxiv.org/content/10.1101/2025.02.26.640110v1)
