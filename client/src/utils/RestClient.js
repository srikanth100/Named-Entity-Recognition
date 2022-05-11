import axios from 'axios';

const get = async (url, request) => {
    try{
        const response = await axios.get(url, request)
        return response.data
    }
    catch(error){
        throw new Error(error)
    }
}

const post = async (url, requestBody) => {
    try{
        const req = {
            data: requestBody
        }
        const headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS, PUT, DELETE',
            'Access-Control-Allow-Headers': 'Origin, Content-Type, Accept, Authorization, X- Request-With'
        }
        const response = await axios.post(url, req, {headers})
        return response.data
    }
    catch(error){
        throw new Error(error)
    }
}

export default { get, post}