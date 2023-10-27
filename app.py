from flask import Flask
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from bson.json_util import dumps
from flask import jsonify,request
app=Flask(__name__)
app.secret_key="secretkey"
app.config['MONGO_URI']="mongodb://localhost:27017/Ecommerce"
mongo=PyMongo(app)

@app.route('/welcome',methods=['GET']) 
def welcome(): 
    return "Welcome to our brand new Ecommerce store E-Kart" 

@app.route('/add',methods=['POST'])
def add_products():
    _json=request.json
    _product_name=_json['product_name']
    _product_id=_json['product_id']
    _product_price=_json['product_price']

    if _product_name and _product_id and _product_price and request.method=='POST':
        id=mongo.db.Products.insert_one({'product_name':_product_name,'product_id':_product_id,'product_price':_product_price})
        resp=jsonify("Products added successfully")
        resp.status_code=200
        return resp
    else:
        return not_found()
    
@app.route('/products',methods=['GET'])
def products():
    products=mongo.db.Products.find()
    resp=dumps(products)
    return resp

@app.route('/product/<id>',methods=['POST'])
def products_id(id):
    product=mongo.db.Products.find_one({"_id":ObjectId(id)})
    resp=dumps(product)
    return resp

@app.route('/delete/<id>',methods=['DELETE'])
def delete_products(id):
    mongo.db.Products.delete_one({"_id":ObjectId(id)})
    resp=jsonify("Products deleted successfully")
    resp.status_code=200
    return resp

@app.route('/update/<id>',methods=['PUT'])
def update_products(id):
    _id=id
    _json=request.json
    _product_name=_json['product_name']
    _product_id=_json['product_id']
    _product_price=_json['product_price']
    if _product_name and _product_id and _product_price and request.method=='PUT':
        mongo.db.Products.update_one({"_id":ObjectId(id['$oid']) if '$oid' in _id else ObjectId(_id)},
                                     {"$set":{"product_name":_product_name,'product_id':_product_id,"product_price":_product_price}})
        resp=jsonify("Product updated successfully")
        resp.status_code=200
        return resp
    else:
        return not_found()
    
@app.errorhandler(404)
def not_found(error=True):
    message={'status':404,
             'message':'Not Found'+'request.url'}
    resp=jsonify(message)
    resp.status_code=404
    return resp

if __name__=="__main__":
    app.run(debug=True,use_reloader=False)