 I ahve some metrics in the pdf
  papers/paper-draft/main.pdf
  but I need to fill out the materials,

  I guess there is a huge gap between the paper and the implementation. I need to do that. Can you please implement it? so that it considers emv, and without emv (just a configuration switch)

  The weights to be added into it

  make sure that all these have been implemented.
  Find the EMV routing algorithm for benchmarking and comparing with Modified hilight.

  Similarly for the case of accidents prevention.

  Important key diff

  Limitations on hilight page 9. (Mention as the key improvement)

  Manual clustering should not be used - concrete proof
  Try to get better results than hilight

  Focus on clustering - try to decrease or increase the cluster size

  Try manually more specific to the

  Db scan i8

  27

  Community detection within clustering

  next

  Elsevier format

  Sorting order- appearance wise

  Naming: abbreviation type iit papers

  Under proposed work:

  Diagramatic architectires

  Importance for emv vehicles -
  Experimentations along with results. Just address all the

  Highlight - the following

  1. Refer the basics for a traffic problem
  2. Show off the 4 adjacent to any number of adjacent roads in an intersections
  3. Specify SUMO with conditions

  Highlight used order of appearance - references sorting
  Weightage change from the value to variable - user preference
  Argue using the papers in results and discussions too
  Results and Discussions to be added
  Remove bold highlights
  AFter the introduction, at the last paragraph - 3 fold and 4 fold contributiondered in this paper. Population is very high.
  The if we need some changes to the current paper main.pdf kindly make it to save content

  Then also accurately implement the mechanism those are all mentioned in the paodebase.
  There are some metrics those are mentioned in the paper. we need to accuratelyctual values.
  Also we need some graphs and images for the following

  1. Training loss of the AC model
  2. Reward functions (overall reward, Reff, Remv, Rfair) all in the same graph,g the variability if hyperparameters are tuned [Image #3] use the ui of the
  image not the metrics or graphical interperation from this.
  3. 2 images on Actual Cologne8 Map, before clustering (show the traffic signalfter clustering (clusters are nothing but traffic intersections with node
  colors such that same node colors represent the intersections belong to the sa_cluster.png and after_cluster.png
  4. Emergency Vehicle routing, use training loss values with_emv.png and withoue shown.

  Use matplotlib and seaborn for the metrics images and show the elements correc

  Also fill the metrics in the table store it in a txt/csv file for easy access.

  Make changes to the corresponding pipeline and train, eval the metrics. write trics/ in the root and just dump all the metrics, resulting images into this.

  For clustering and its result, try to take a screen capture using SUMO if not