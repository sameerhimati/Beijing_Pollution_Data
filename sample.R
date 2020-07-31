library(ISLR)
library(splines)
library(gam)
attach(d.train)

# loop.vector <- 1:110
# for (i in loop.vector) { # Loop over loop.vector
#   
#   # store data in column.i as x
#   x <- d.train[,i]
#   
#   # Plot histogram of x
#   plot(x,y, main = i)
#   linear <- lm(y~x)
#   
#   summary(linear)
#   summary(linear)$coefficients[,4]
# }

gam1 = lm(y~s(X5, 1)+s(X6)+X10+X11+s(X12)+s(X14)+X13+X15+X16+X17+X19+s(X4)+X39+X63+X37+X51+X62+X69+s(X93, 6)+X31+s(X23)+X18+X9,data=d.train)
plot.Gam(gam1, se=TRUE, col="red")
