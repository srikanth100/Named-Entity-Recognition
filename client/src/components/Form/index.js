import React from 'react';
import RestClient from '../../utils/RestClient';
import Loader from '../Loader'
import DisplayInfo from '../DisplayInfo'
import TextField from '@material-ui/core/TextField';
import { makeStyles } from '@material-ui/core/styles';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Button from '@material-ui/core/Button';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import './FormComponent.css'
import VerticalBar from '../VerticalBar';

const useStyles = makeStyles((theme) => ({
    formControl: {
        margin: theme.spacing(1),
        minWidth: 120,
    },
    root: {
        '& .MuiTextField-root': {
            display: 'flex',
            margin: theme.spacing(1),
            width: '110ch',
            'margin-top': '20px'
        },
    },
}));

export default function FormPropsTextFields() {
    const classes = useStyles();

    const apiErrorMessage = "Something went wrong.Please try again."

    const endpointDictionary = {
        BERTCRF: 'http://localhost:1103/pred',
        BiLSTMCRF: 'http://localhost:1103/pred_bilstm_crf',
        BERT: 'http://localhost:1103/pred_bert',
        CRF: 'http://localhost:1103/pred_crf',
        BiLSTM: 'http://localhost:1103/pred_bilstm',
        Spacy: 'http://localhost:1103/pred_spacy'

    }

    const parse = (data) => {
        let countArr = data.map(x => x.count)
        return countArr
    }

    const [model, setModel] = React.useState('');
    const [submitClicked, setSubmitClick] = React.useState(false);
    const [isLoader, setLoader] = React.useState(false);
    const [textValue, setTextValue] = React.useState('');
    const [response, setApiResponse] = React.useState([]);
    const [errorMessage, setErrorMessage] = React.useState('');
    const [countList, setcountList] = React.useState([]);

    const handleChange = (event) => {
        setModel(event.target.value);
    };

    const getResponse = async () => {
        var data = await Promise.all(Object.values(endpointDictionary).map(url => RestClient.post(url, textValue)))
        setApiResponse(data.map(x => x.data))
        setcountList(parse(data))
    }

    // const getResponse1 = async () => {
    //     var listurl= ['https://api.agify.io/?name=asaa','https://api.agify.io/?name=qwerty','https://api.agify.io/?name=asaaasd']
    //     var data = await Promise.all( listurl.map((url) => RestClient.get(url)))
    //     console.log('data',data)
    //     return data
    // }


    const getNERC = async () => {
        try {
            let res
            if (model == 'All') {
                res = await getResponse()
                setLoader(false)
            }
            else {
                res = await RestClient.post(endpointDictionary[model], textValue)
                setApiResponse([res.data])
                setLoader(false)
            }
        }
        catch {
            setErrorMessage(apiErrorMessage)
            setLoader(false)
        }
    };

    const handleSubmit = () => {
        setApiResponse({})
        setErrorMessage('')
        setSubmitClick(true)
        if (textValue.length == 0) {
            setErrorMessage('No input given')
        }
        else if (model.length == 0) {
            setErrorMessage('No model selected')
        }
        else {
            setLoader(true)
            getNERC() // API call
        }
        // setSubmitClick(false)
    }

    const handleTextChange = (event) => {
        setTextValue(event.target.value)
    }

    return (
        <React.Fragment>
            <form className={classes.root} noValidate autoComplete="off">
            <div className="selectgroup">
                    <FormControl variant="filled" className={classes.formControl}>
                        <InputLabel id="demo-simple-select-filled-label">Model</InputLabel>
                        <Select
                            labelId="demo-simple-select-filled-label"
                            id="demo-simple-select-filled"
                            value={model}
                            onChange={handleChange}
                        >
                            <MenuItem value="">
                                <em>None</em>
                            </MenuItem>
                            <MenuItem value="BERT">BERT</MenuItem>
                            <MenuItem value="CRF">CRF</MenuItem>
                            <MenuItem value="BiLSTMCRF">BiLSTM+CRF</MenuItem>
                            <MenuItem value="BERTCRF">BERT+CRF</MenuItem>
                            <MenuItem value="BiLSTM">BiLSTM</MenuItem>
                            <MenuItem value="Spacy">Spacy</MenuItem>
                            <MenuItem value="All">All</MenuItem>
                        </Select>
                    </FormControl>
                </div>
                <div className="textfield">
                    <TextField
                        id="outlined-helperText"
                        label="Enter the text"
                        defaultValue=""
                        helperText=""
                        variant="outlined"
                        onChange={handleTextChange}
                    />
                </div>
            </form>
            <div className="button">
                <Button variant="contained" color="secondary" onClick={handleSubmit}>
                    ENTER
                </Button>
            </div>
            {isLoader && <Loader />}
            {submitClicked && !isLoader && (response.length != 0 || errorMessage.length != 0) &&
                <DisplayInfo words={response} errorMessage={errorMessage} model={model} endpointKeys={Object.keys(endpointDictionary)}/>}
            {model == 'All' && submitClicked && !isLoader && countList.length != 0 && <VerticalBar countList={countList} />}
        </React.Fragment>
    );
}