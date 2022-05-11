import './DisplayInfo.css';

function DisplayInfo(props) {
    var model = props.model == 'All' ? props.endpointKeys : [props.model]
    function SuccessResponse() {
       return <>
        {props.words.map((word,index) =>
         (<div className="displayInfo">
            Named Entity Recognition {model[index]}:
            <div className="displayPar">
                {word.map(wordArray => (<div className="wordUnit"><span className="wordKey">{wordArray[0]}</span> <span className="wordValue">{wordArray[1].split("-")[1]}</span></div>)
                )}
            </div>
        </div>))}
        </>
    }

    function ErrorResponse() {
        return <div className="displayInfo">
                    {props.errorMessage}
                </div>;
    }

    return (
        !props.errorMessage && props.words && props.words.length !=0 ? SuccessResponse(): ErrorResponse()
    );
}

export default DisplayInfo;
