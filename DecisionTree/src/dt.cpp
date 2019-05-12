#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <math.h>
#include <cstring>
#include <limits>
#include <fstream>

using namespace std;

struct Node{
    int splitIndex;
    string arrivedString;
    string label;
    vector<Node> children;
    Node():splitIndex(-1),label(""){};
};


double computeInfo(vector<vector<string>>& samples,int index){
    map<string,int> mp;//mp:attribute map ,labelmp:label map
    int numClasses = 0;
    vector<vector<int>> count;
    for(auto &sample:samples) {
        if (mp.count(sample[index]) == 0) {
            mp[sample[index]] = numClasses++;
            vector<int> tmp(2);// yes or no
            count.push_back(tmp);
        }
        if(strcmp(sample.back().c_str(),"yes")==0)
            count[mp[sample[index]]][0]++;
        else
            count[mp[sample[index]]][1]++;
    }

    double info = 0.;
    double eps = 1e-8;
    for(int i=0;i<count.size();i++){
        int yes = count[i][0];
        int no = count[i][1];
        double total = yes + no;
        info -= (yes/total*log2(yes/total+eps)+no/total*log2(no/total+eps));
    }
    return info/numClasses;//c45 info/numOfSubNode
}

int getSplitIndex(vector<vector<string>> &samples,vector<int> remain_index){
    double mininfo=std::numeric_limits<double>::max();
    int splitIndex = -1;
    for (auto i:remain_index) {
        double tmp = computeInfo(samples,i);
        if(mininfo>tmp){
            mininfo = tmp;
            splitIndex = i;
        }
    }
    return splitIndex;
}

bool isAllSame(vector<vector<string>> &samples,Node &root){
    if(samples.size()<=1){
        root.label = samples[0].back();
        return true;
    }

    string s0 = samples[0].back();
    for (int i = 1; i <samples.size() ; ++i) {
        string s1 = samples[i].back();
        if(s0!=s1)
            return false;
    }
    root.label = s0;
    return true;
}

vector<int> deleteIndex(vector<int> remain_index,int splitIndex){
    vector<int> ret;
    for(auto i:remain_index){
        if(i!=splitIndex)
            ret.push_back(i);
    }
    return ret;
}

void generateTree(vector<vector<string>> &samples,Node &root,vector<int> remain_index){
    bool isSame = isAllSame(samples,root);
    if(isSame){
        root.label = samples[0].back();
        return ;
    }
    int splitIndex = getSplitIndex(samples,remain_index);
    if (splitIndex==-1)
        return ;
    root.splitIndex = splitIndex;
    //generate subtree
    map<string,int> mp;//index string to int
    vector<vector<vector<string>>> _samples;
    int numClasses = 0;
    for(auto &sample:samples){
        string indexString = sample[splitIndex];
        if(mp.count(indexString)==0) {
            mp[indexString] = numClasses++;
            Node node;
            node.arrivedString = indexString;
            root.children.push_back(node);
            vector<vector<string>> tmp;
            _samples.push_back(tmp);
        }
        _samples[mp[indexString]].push_back(sample);
    }

    remain_index = deleteIndex(remain_index,splitIndex);
    for(int i=0;i<_samples.size();i++){
        generateTree(_samples[i],root.children[i],remain_index);
    }
}

vector<vector<string>> readDataFromFile(string filename="../data.txt"){
    ifstream in(filename,ios::in);
    if (! in.is_open())
    { cout << "Error opening file"; exit (1); }
    int i=0;
    vector<string> tmp;
    vector<vector<string>> ret;
    while(!in.eof()){
        string buff;
        in>>buff;
        tmp.push_back(buff);
        if(i>=3){
            if((i-3)%5==0){
                ret.push_back(tmp);
                tmp.clear();
            }
        }
        i++;
    }
    return ret;
}

void show(Node root,int depth,vector<string> attribute){
    if(root.arrivedString!="")
        cout<<string(depth,' ')<<root.arrivedString<<endl;
    if(root.splitIndex!=-1)
        cout<<string(depth+1,' ')<<attribute[root.splitIndex]<<endl;
    for(Node n:root.children){
        show(n,depth+2,attribute);
    }
    if(root.label!="")
        cout<<string(depth+3,' ')<<root.label<<endl;
}

string predict(Node root,vector<string> data){
    while(root.label==""){
        for(auto child:root.children){
            if(child.arrivedString==data[root.splitIndex]){
                root = child;
                break;
            }
        }
    }
    return root.label;
}

int main(){
    vector<vector<string>> data=readDataFromFile();
    vector<string> attribute=data[0];
    data.erase(data.begin());
    vector<int> remain_index;
    for (int i = 0; i <data[0].size() -1 ; ++i) {
        remain_index.push_back(i);
    }
    Node root;
    generateTree(data,root,remain_index);
    //show
    show(root,0,attribute);
    //test
    for(auto d:data){
        cout<<predict(root,d)<<endl;
    }
    return 0;
}
