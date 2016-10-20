FROM ipython/scipyserver
MAINTAINER Jean-Francois Puget <j-f.puget@fr.ibm.com>
EXPOSE 9000
RUN cp /usr/local/bin/pip2 /usr/local/bin/pip
 
