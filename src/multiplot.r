# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
library(ggplot2)
library(grid)

library(repr)
options(repr.plot.width=20, repr.plot.height=8)

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {


  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

hist_with_kde <- function (feature, feature_name) {
    plot <- qplot(feature, geom="histogram", bins=200, alpha=I(.4), y = ..density..)+
        geom_vline(aes(xintercept=mean(feature, rm.na=T)), color="red", linetype="dashed", size=1)+
        geom_vline(aes(xintercept=median(feature)), color="blue", linetype="dashed", size=1)+
        labs(x=feature_name, y="density")+
        geom_density()
    return(plot)
}

hist_with_kde_numerical_by_category <- function (numerical_feature, categorical_feature, num_feature_name, cat_feature_name) {
    plot <- qplot(numerical_feature, geom="histogram", bins=200, alpha=I(.4), 
                  y = ..density.., fill=categorical_feature) +
    labs(x=num_feature_name, y="density")+
    geom_density(alpha=0.2)
    return(plot)
}