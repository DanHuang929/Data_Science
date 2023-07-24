#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cmath>
#include <iomanip>
using namespace std;

vector<int> stringTOnum1(string);
vector<vector<int> > generateNextC(vector<vector<int> >, int);
vector<int> getElement(vector<vector<int> >);
vector<vector<int> > join(vector<vector<int> >, int);
vector<vector<int> > pruning (vector<vector<int> >, vector<vector<int> >);
float calculate_support(vector<int>);
vector<vector<int> > generateL(vector<vector<int> >);
void apriori() ;


vector<vector<int> > transaction;
vector<vector<int> > frequent_pattern;
vector<float> support;
float min_support;

vector<int> stringTOnum1(string s)
{
	bool temp=false;		
	int data=0;				
	vector<int> arr;				


	for(int i=0;i<s.length();i++){
		while((s[i]>='0')&&(s[i]<='9')){
			temp=true;		
			data*=10;
			data+=(s[i]-'0');		
			i++;
		}

		if(temp){
			arr.push_back(data);	
			data=0;		
			temp=false;	
		}
	}
	return arr;
}


void apriori() {
	int step=0;
	vector<vector<int> > C, L;
	while(true) {
		C = generateNextC(L, step);

		if(C.size()==0) break;
		step++;
		L = generateL(C);
	
	}
}


vector<vector<int> > generateNextC(vector<vector<int> > L, int step) {
	if(step==0) {
		vector<vector<int> > ret;
		vector<int> element = getElement(transaction);
		for (int i = 0; i < element.size(); i++)
			ret.push_back(vector<int>(1, i));
		
		return ret;
	} else {
		return pruning(join(L, step), L);
	}
}

vector<int> getElement(vector<vector<int> > itemset) {
        vector<int> element;
        set<int> s;
        for (int i=0; i<itemset.size(); i++)
			for (int j=0; j<itemset[i].size(); j++)
				s.insert(itemset[i][j]);
        for(auto iter=s.begin(); iter != s.end(); iter++) element.push_back(*iter);
        return element;
    }

vector<vector<int> > join(vector<vector<int> > L, int step) {
	vector<vector<int> > ret;
	for(int i=0;i<L.size();i++){
		for(int j=i+1;j<L.size(); j++) {
			int k;
			for(k=0;k<step-1; k++) {
				if(L[i][k] != L[j][k]) break;
			}
			if(k == step-1) {
				vector<int> tmp;
				for(int k=0;k<step-1; k++) {
					tmp.push_back(L[i][k]);
				}
				int a = L[i][step-1];
				int b = L[j][step-1];
				if(a>b) swap(a,b);
				tmp.push_back(a), tmp.push_back(b);
				ret.push_back(tmp);
			}
		}
	}
	return ret;
}

vector<vector<int> > pruning (vector<vector<int> > joined, vector<vector<int> > L) {
	vector<vector<int> > ret;
	
	set<vector<int> > lSet;
	for (int i=0; i<L.size(); i++)
		lSet.insert(L[i]);

	for (int j=0; j<joined.size(); j++){
		int i;
		for(i=0;i<joined[j].size();i++){
			vector<int> tmp = joined[j];
			tmp.erase(tmp.begin()+i);
			if(lSet.find(tmp) == lSet.end()) {
				break;
			}
		}
		if(i==joined[j].size()){
			ret.push_back(joined[j]);
		}
	}
	return ret;
}

float calculate_support(vector<int> item) {
	int ret = 0;
	for(int k=0; k<transaction.size(); k++){
		int i, j;
		if(transaction[k].size() < item.size()) continue;
		for(i=0, j=0; i < transaction[k].size();i++) {
			if(j==item.size()) break;
			if(transaction[k][i] == item[j]) j++;
		}
		if(j==item.size()){
			ret++;
		}
	}
	return (float)ret/transaction.size();
}

vector<vector<int> > generateL(vector<vector<int> > C) {
	vector<vector<int> > ret;
	for(int i=0; i<C.size(); i++){
		float S = calculate_support(C[i]);
		// cout<<S<<" ###"<<endl;
		if(S < min_support) continue;
		else{
			ret.push_back(C[i]);
			frequent_pattern.push_back(C[i]);
			support.push_back(S);
		}
		
	}
	return ret;
}



int main(int argc, char *argv[]) {
	
	min_support = stod(argv[1]);
	string input_file = argv[2];
	string output_file = argv[3];
      
    string s;	
	ifstream ifs;
	ifs.open(input_file);
	while(getline(ifs, s)){
        transaction.push_back(stringTOnum1(s));
	}
	ifs.close();	

	apriori();

	ofstream ofs;
	ofs.open(output_file);
	for(int i=0; i<frequent_pattern.size(); i++){
        for(int j=0; j<frequent_pattern[i].size()-1; j++){
			ofs<<frequent_pattern[i][j]<<",";
        }
		ofs<<frequent_pattern[i][frequent_pattern[i].size()-1]<<":";
		ofs<<fixed<<setprecision(4)<<support[i]<<endl;
    }
	ofs.close();	

    return 0;
}