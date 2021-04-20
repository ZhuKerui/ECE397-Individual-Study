#! /bin/bash

# Remove \n string, $, \\, ---, contents in [], contents in (), 
# change '-' into ' - ', remove extra space and change all char to lower case
sed 's/\\n/ /g;s/\$//g;s/\\/ /g;s/---*/, /g;s/([^)]*)//g;s/{[^)]*}//g;s/-/ - /g' $1 | tr -s [:space:] | tr '[:upper:]' '[:lower:]' > $2